# util/algo/sesd.py
import torch
import torch.nn as nn
import numpy as np
import os
import traceback
from tqdm import tqdm
import lpips
import fastmri
from util.algo.utils import compute_metrics as compute_metrics_util, log_metrics_to_tensorboard, ESWithWMV

# ═══════════════════════════════════════════════════════════════════════
# Measurement Operator Abstraction Layer
# ═══════════════════════════════════════════════════════════════════════

class MeasurementOperator:
    """Base class for measurement operators in SESD."""
    def forward(self, x, mask=None):
        raise NotImplementedError
    
    def project(self, x, y, mask=None, alpha=None):
        """
        Projects x onto the measurement manifold defined by y.
        Returns the data consistency update v_k.
        """
        raise NotImplementedError


class StandardOperator(MeasurementOperator):
    """
    Standard operator for deblurring, super-resolution, inpainting tasks.
    """
    def __init__(self, operator_module, device, mu=1.0):
        self.operator = operator_module
        self.device = device
        self.mu = torch.tensor(mu, device=device)

    def forward(self, x, mask=None):
        return self.operator.forward(x, mask=mask) if mask is not None else self.operator.forward(x)

    def project(self, x, y, mask=None, alpha=None):
        """Gradient descent based projection for general linear/nonlinear problems."""
        with torch.enable_grad():
            x_in = x.detach().clone().requires_grad_(True)
            Ax = self.forward(x_in, mask)
            
            # Use MSE-like gradient for stability
            loss = torch.linalg.norm(y - Ax)
            grad = torch.autograd.grad(loss, x_in)[0]
            
            v_k = x - self.mu * grad
        return v_k


class MRIOperator(MeasurementOperator):
    """
    MRI specific operator handling multi-coil k-space data.
    """
    def __init__(self, device, k_under, mask_dc, csm, img_min, img_max):
        self.device = device
        self.k_under = k_under
        self.mask_dc = mask_dc
        self.csm = csm
        self.img_min = img_min
        self.img_max = img_max
        
        # Internal modules
        self.dc_layer = DC_layer_CSM().to(device)
        self.mri_fwd = mri_forward().to(device)

    def forward(self, x, mask=None):
        return self.mri_fwd(x, self.mask_dc, self.csm, self.img_min, self.img_max)

    def project(self, x, y, mask=None, alpha=None):
        """Data consistency layer projection."""
        # x is [B, C, H, W]. MRI code uses x[:, 0:1, ...]
        x_single = x[:, 0:1, :, :]
        v_k_single = self.dc_layer(x_single, self.k_under, self.mask_dc, self.csm, self.img_min, self.img_max)
        
        # Broadcast back to C channels if necessary (e.g. for model input)
        v_k = v_k_single.repeat(1, x.shape[1], 1, 1).contiguous()
        return v_k


# ═══════════════════════════════════════════════════════════════════════
# Core Algorithm: SESD with ALES
# ═══════════════════════════════════════════════════════════════════════

def SESD_Core(
    model, sampler, measurement_cond_fn, ref_img, y_n,
    model_config, operator: MeasurementOperator, fname,
    iter_step, iteration, denoiser_step, lr, out_path,
    loss_fn_alex, device, mask_sampling=None, random_seed=None,
    writer=None, img_index=None,
    # ALES parameters
    use_ales=True,
    ales_window_size=10,
    ales_var_threshold=1e-3,
    ales_alpha=1e-3,
    ales_patience=20,
    ales_min_epochs=30
):
    """
    SESD: Score Evolved Shortcut Diffusion with ALES (Adaptive Local Early Stopping).
    
    ALES Parameters:
        use_ales: Whether to enable ALES early stopping
        ales_window_size: Window size W for time-weighted variance
        ales_var_threshold: Variance threshold δ_v
        ales_alpha: Loss threshold α
        ales_patience: Patience parameter P
        ales_min_epochs: Minimum iterations E_min before early stopping
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)

    # ─────────────────────────────────────────────────────────────────
    # Logging configuration to TensorBoard
    # ─────────────────────────────────────────────────────────────────
    algo_name = "SESD"
    if writer and img_index is not None:
        config_text = (
            f'Algorithm: {algo_name}\n'
            f'Iterations: {iteration}\n'
            f'LR: {lr}\n'
            f'Shortcut Step t*: {iter_step}\n'
            f'Total Steps T: {denoiser_step}\n'
            f'ALES Enabled: {use_ales}\n'
        )
        if use_ales:
            config_text += (
                f'ALES Window Size W: {ales_window_size}\n'
                f'ALES Var Threshold δ_v: {ales_var_threshold}\n'
                f'ALES Alpha α: {ales_alpha}\n'
                f'ALES Patience P: {ales_patience}\n'
                f'ALES Min Epochs E_min: {ales_min_epochs}\n'
            )
        writer.add_text(f'{algo_name}/Image_{img_index}/Config', config_text, 0)

    # ─────────────────────────────────────────────────────────────────
    # 1. Initialize Z at t* (Shortcut)
    # ─────────────────────────────────────────────────────────────────
    Z_channels = 3  # Standard RGB
    Z = torch.randn((1, Z_channels, model_config['image_size'], model_config['image_size']), device=device)
    
    current_state = Z
    with torch.no_grad():
        # Shortcut: sampling from T down to t* (iter_step)
        for i in range(denoiser_step - 1, iter_step - 1, -1):
            t_val = torch.tensor([i] * Z.shape[0], device=device)
            current_state, _ = sampler.p_sample(
                model=model, x=current_state, t=t_val,
                measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask=mask_sampling
            )
    
    initial_shortcut_state = current_state.detach().clone()
    
    # ─────────────────────────────────────────────────────────────────
    # 2. Optimization Setup
    # ─────────────────────────────────────────────────────────────────
    x_opt = initial_shortcut_state.requires_grad_(True)
    
    # Learnable balancing parameter λ (renamed from α in rebuttal for clarity)
    lambda_param = torch.tensor(0.5, requires_grad=True, device=device) 
    
    optimizer = torch.optim.Adam([
        {'params': x_opt, 'lr': lr},
        {'params': lambda_param, 'lr': lr * 0.1}  # Smaller LR for λ
    ])
    
    data_fidelity_loss_fn = nn.L1Loss().to(device)
    
    # ─────────────────────────────────────────────────────────────────
    # 3. ALES Early Stopper Initialization
    # ─────────────────────────────────────────────────────────────────
    if use_ales:
        early_stopper = ESWithWMV(
            window_size=ales_window_size,
            var_threshold=ales_var_threshold,
            alpha=ales_alpha,
            patience=ales_patience,
            min_epochs=ales_min_epochs,
            verbose=True
        )
    
    best_psnr = -float('inf')
    best_sample = None
    best_metrics = None
    best_epoch = 0
    
    psnrs_log = []
    
    # ─────────────────────────────────────────────────────────────────
    # 4. Optimization Loop (with ALES)
    # ─────────────────────────────────────────────────────────────────
    pbar = tqdm(range(iteration), desc=f"SESD Opt. Img {img_index or ''}")
    for epoch in pbar:
        model.eval()
        optimizer.zero_grad()
        
        # ═══ A. Denoise from current x_{t*} to x_0 (approx) ═══
        denoised_state = x_opt
        for i in range(iter_step - 1, -1, -1):
             t_val = torch.tensor([i] * denoised_state.shape[0], device=device)
             denoised_state, _ = sampler.p_sample(
                 model=model, x=denoised_state, t=t_val,
                 measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask=mask_sampling
             )
        
        # ═══ B. Data Consistency via Operator Projection ═══
        v_k = operator.project(denoised_state, y_n, mask=mask_sampling)
        
        # ═══ C. Fusion (using λ instead of α) ═══
        current_lambda = torch.sigmoid(lambda_param)
        x_k_fusion = current_lambda * denoised_state + (1 - current_lambda) * v_k
        
        # ═══ D. Loss Calculation ═══
        est_measurement = operator.forward(x_k_fusion, mask=mask_sampling)
        loss = data_fidelity_loss_fn(est_measurement, y_n)
        
        # ═══ E. Backpropagation ═══
        loss.backward()
        optimizer.step()
        
        # ═══ F. Metrics Logging & ALES Check ═══
        with torch.no_grad():
            sample_eval = x_k_fusion
            if ref_img.shape[1] == 1 and x_k_fusion.shape[1] == 3:
                sample_eval = x_k_fusion[:, 0:1, :, :]
                
            curr_metrics = compute_metrics_util(
                sample=sample_eval, ref_img=ref_img, device=device, loss_fn_alex=loss_fn_alex
            )
            curr_psnr = curr_metrics.get('psnr', float('nan'))
            psnrs_log.append(curr_psnr)
            
            pbar.set_postfix({
                'loss': loss.item(), 
                'λ': current_lambda.item(), 
                'psnr': curr_psnr
            })
            
            # Update best sample
            if not np.isnan(curr_psnr) and curr_psnr > best_psnr:
                best_psnr = curr_psnr
                best_sample = x_k_fusion.detach().clone()
                best_metrics = curr_metrics.copy()
                best_epoch = epoch
                
            # TensorBoard logging
            if writer and img_index is not None:
                log_metrics_to_tensorboard(writer, {
                    'Loss': loss.item(), 
                    'PSNR': curr_psnr, 
                    'SSIM': curr_metrics.get('ssim'),
                    'Lambda': current_lambda.item()
                }, epoch, img_index, prefix=f'{algo_name}/Epoch')
            
            # ═══ ALES Early Stopping Check ═══
            if use_ales:
                should_stop = early_stopper(epoch, x_k_fusion, loss.item())
                if should_stop:
                    print(f"\n🛑 ALES triggered early stopping at epoch {epoch+1}")
                    if writer and img_index is not None:
                        writer.add_text(
                            f'{algo_name}/Image_{img_index}/ALES_Stop',
                            f'ALES stopped at epoch {epoch+1}', epoch
                        )
                    break

    if best_sample is None: 
        best_sample = x_k_fusion.detach()
    
    # ─────────────────────────────────────────────────────────────────
    # 5. Save Results
    # ─────────────────────────────────────────────────────────────────
    save_subdir = 'sesd_results'
    _save_algo_image(
        best_sample, 
        os.path.join(out_path, f'recon_{save_subdir}', fname), 
        is_mri_grayscale=(ref_img.shape[1]==1)
    )
    
    print(f"SESD Final {fname}: Best PSNR {best_psnr:.4f} at epoch {best_epoch}")
    
    return best_sample, best_metrics, {'psnrs': psnrs_log}


# ═══════════════════════════════════════════════════════════════════════
# Compatibility Wrappers
# ═══════════════════════════════════════════════════════════════════════

def SESD(
    model, sampler, measurement_cond_fn, ref_img, y_n, device, 
    model_config, measure_config, operator, fname,
    iter_step=3, iteration=300, denoiser_step=10, lr=0.02, 
    out_path='./outputs/', mask=None, random_seed=None, 
    writer=None, img_index=None, loss_fn_alex=None,
    # ALES parameters (defaults match paper Table VI)
    use_ales=True,
    ales_window_size=10,
    ales_var_threshold=1e-3,
    ales_alpha=1e-3,
    ales_patience=20,
    ales_min_epochs=30,
    **kwargs
):
    """
    Standard SESD wrapper for super-resolution, inpainting, and deblurring.
    """
    if loss_fn_alex is None: 
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    # Create standard operator
    std_operator = StandardOperator(operator, device)
    
    return SESD_Core(
        model, sampler, measurement_cond_fn, ref_img, y_n,
        model_config, std_operator, fname,
        iter_step, iteration, denoiser_step, lr, out_path,
        loss_fn_alex, device, mask_sampling=mask, random_seed=random_seed,
        writer=writer, img_index=img_index,
        use_ales=use_ales, 
        ales_window_size=ales_window_size,
        ales_var_threshold=ales_var_threshold,
        ales_alpha=ales_alpha,
        ales_patience=ales_patience,
        ales_min_epochs=ales_min_epochs
    )


def SESD_MRI(
    model, sampler, measurement_cond_fn, ref_img, y_n, 
    k_under, mask_dc, csm, img_min, img_max,
    device, model_config, measure_config, operator, fname,
    iter_step=3, iteration=300, denoiser_step=10, lr=0.02,
    out_path='./outputs/', random_seed=None,
    writer=None, img_index=None, loss_fn_alex=None,
    # ALES parameters
    use_ales=True,
    ales_window_size=10,
    ales_var_threshold=1e-3,
    ales_alpha=1e-3,
    ales_patience=20,
    ales_min_epochs=30,
    **kwargs
):
    """
    MRI-specific SESD wrapper for multi-coil k-space reconstruction.
    """
    if loss_fn_alex is None: 
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    # Create MRI operator
    mri_operator = MRIOperator(device, k_under, mask_dc, csm, img_min, img_max)
    
    return SESD_Core(
        model, sampler, measurement_cond_fn, ref_img, y_n,
        model_config, mri_operator, fname,
        iter_step, iteration, denoiser_step, lr, out_path,
        loss_fn_alex, device, mask_sampling=mask_dc, random_seed=random_seed,
        writer=writer, img_index=img_index,
        use_ales=use_ales,
        ales_window_size=ales_window_size,
        ales_var_threshold=ales_var_threshold,
        ales_alpha=ales_alpha,
        ales_patience=ales_patience,
        ales_min_epochs=ales_min_epochs
    )


# Legacy aliases for backward compatibility
acce_RED_diff = SESD
acce_RED_diff_mri = SESD_MRI


# ═══════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════

def _save_algo_image(tensor_data, file_path, is_kernel=False, is_mri_grayscale=False):
    """Save tensor image to file with proper format handling."""
    import matplotlib.pyplot as plt
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        img = tensor_data.detach().cpu()
        if img.dim() == 4: 
            img = img[0]
        
        if is_mri_grayscale: 
             if img.dim() == 3 and img.shape[0] == 3: 
                 img = img[0] 
             if img.dim() == 3: 
                 img = img.squeeze(0)
             plt.imsave(file_path, img.numpy(), cmap='gray')
        else:
             if img.dim() == 3 and img.shape[0] == 1: 
                 img = img.squeeze(0)
                 plt.imsave(file_path, img.numpy(), cmap='gray')
             else:
                 img = img.permute(1, 2, 0).numpy()
                 img = (img + 1) / 2
                 img = np.clip(img, 0, 1)
                 plt.imsave(file_path, img)
    except Exception as e:
        print(f"Error saving {file_path}: {e}")


# ═══════════════════════════════════════════════════════════════════════
# MRI Utilities (Minimal Set Required for MRIOperator)
# ═══════════════════════════════════════════════════════════════════════

def kspace2rss(kspace_data_real): 
    """
    Convert multi-coil k-space data to RSS (Root Sum of Squares) image.
    
    Args:
        kspace_data_real: k-space data with real representation (B, num_coils, H, W, 2)
    
    Returns:
        RSS image (B, 1, H, W)
    """
    kspace_complex = torch.view_as_complex(kspace_data_real.contiguous())
    image_space_coils = fastmri.ifft2c(kspace_complex) 
    abs_coil_images = fastmri.complex_abs(image_space_coils)
    rss_image = fastmri.rss(abs_coil_images, dim=1) 
    return rss_image.unsqueeze(1) 


def rss_to_kspace(rss_image_normalized, csm, img_min, img_max):
    """
    Convert RSS image back to multi-coil k-space data.
    
    Args:
        rss_image_normalized: Normalized RSS image (B, 1, H, W) in [-1, 1]
        csm: Coil sensitivity maps (B, num_coils, H, W, 2)
        img_min: Original RSS min values for denormalization
        img_max: Original RSS max values for denormalization
    
    Returns:
        k-space data (B, num_coils, H, W, 2)
    """
    rss_image = ((rss_image_normalized.squeeze(1) + 1.0) / 2.0) * \
                (img_max.view(-1,1,1) - img_min.view(-1,1,1)) + img_min.view(-1,1,1)
    csm_complex = torch.view_as_complex(csm.contiguous())
    coil_images = csm_complex * rss_image.unsqueeze(1)
    kspace = fastmri.fft2c(coil_images)
    return torch.view_as_real(kspace.contiguous())


class DC_layer_CSM(nn.Module):
    """Data Consistency layer using Coil Sensitivity Maps."""
    def __init__(self): 
        super().__init__()
        
    def forward(self, x_rss, k_under, mask, csm, img_min, img_max):
        """
        Apply data consistency correction in k-space.
        
        Args:
            x_rss: Input RSS image (B, 1, H, W), normalized to [-1, 1]
            k_under: Undersampled k-space data (B, num_coils, H, W, 2)
            mask: Sampling mask
            csm: Coil sensitivity maps (B, num_coils, H, W, 2)
            img_min: Original RSS min for denormalization
            img_max: Original RSS max for denormalization
        
        Returns:
            Corrected RSS image (B, 1, H, W), renormalized to [-1, 1]
        """
        k_est = rss_to_kspace(x_rss, csm, img_min, img_max)
        mask_bool = mask.squeeze(-1)
        k_dc = (1 - mask_bool.float()) * torch.view_as_complex(k_est.contiguous()) + \
               mask_bool.float() * torch.view_as_complex(k_under.contiguous())
        rss_new = kspace2rss(torch.view_as_real(k_dc))
        denom = (img_max.view(-1,1,1,1) - img_min.view(-1,1,1,1) + 1e-7)
        rss_norm = ((rss_new - img_min.view(-1,1,1,1)) / denom) * 2.0 - 1.0
        return torch.clamp(rss_norm, -1.0, 1.0)


class mri_forward(nn.Module):
    """MRI forward operator: image → undersampled k-space → image."""
    def __init__(self): 
        super().__init__()
        
    def forward(self, x_rss, mask, csm, img_min, img_max):
        """
        Forward MRI operator with undersampling simulation.
        
        Args:
            x_rss: Input RSS image (B, 1, H, W), normalized to [-1, 1]
            mask: Sampling mask
            csm: Coil sensitivity maps
            img_min: Original RSS min
            img_max: Original RSS max
        
        Returns:
            Undersampled RSS image (B, 1, H, W), renormalized to [-1, 1]
        """
        k_full = rss_to_kspace(x_rss, csm, img_min, img_max)
        mask_to_apply = mask
        if mask_to_apply.dim() == 2: 
            mask_to_apply = mask_to_apply.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        if mask_to_apply.shape[-1] == 1: 
            mask_to_apply = mask_to_apply.squeeze(-1)
        k_under = torch.view_as_complex(k_full.contiguous()) * mask_to_apply.unsqueeze(1)
        rss_under = kspace2rss(torch.view_as_real(k_under))
        denom = (img_max.view(-1,1,1,1) - img_min.view(-1,1,1,1) + 1e-7)
        rss_norm = ((rss_under - img_min.view(-1,1,1,1)) / denom) * 2.0 - 1.0
        return torch.clamp(rss_norm, -1.0, 1.0)
