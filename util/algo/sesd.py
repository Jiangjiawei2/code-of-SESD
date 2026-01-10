# util/algo/sesd.py
"""
SESD: Score Evolved Shortcut Diffusion with ALES
For standard inverse problems: super-resolution, deblurring, inpainting
"""
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from util.algo.utils import ESWithWMV, log_metrics_to_tensorboard


def sesd(
    model, sampler, measurement_cond_fn, ref_img, y_n, device, 
    model_config, measure_config, operator, fname,
    iter_step=3, iteration=300, denoiser_step=10, lr=0.02,
    out_path='./outputs/', mask=None, random_seed=None,
    writer=None, img_index=None,
    # ALES parameters (Algorithm 2 in paper)
    use_ales=True,
    window_size=10,        # W: window size for time-weighted variance
    var_threshold=1e-3,    # δ_v: variance threshold
    loss_threshold=1e-3,   # α: loss threshold  
    patience=20,           # P: patience for early stopping
    min_epochs=30,         # E_min: minimum iterations before stopping
    **kwargs
):
    """
    SESD: Score Evolved Shortcut Diffusion with ALES
    
    This is the standard version for image inverse problems.
    
    Args:
        model: Pretrained diffusion model
        sampler: Diffusion sampler (e.g., DDPM, DDIM)
        measurement_cond_fn: Measurement conditioning function
        ref_img: Reference/ground truth image
        y_n: Degraded measurement
        device: Computation device
        model_config: Model configuration dict
        measure_config: Measurement configuration dict
        operator: Forward operator A (e.g., blur, downsample, mask)
        fname: Filename for saving results
        iter_step: Shortcut timestep t* (default: 3)
        iteration: Max optimization iterations (default: 300)
        denoiser_step: Total diffusion steps T (default: 10)
        lr: Learning rate (default: 0.02)
        out_path: Output directory
        mask: Optional mask for inpainting
        random_seed: Random seed for reproducibility
        writer: TensorBoard writer
        img_index: Image index for logging
        use_ales: Enable ALES early stopping
        window_size: ALES parameter W
        var_threshold: ALES parameter δ_v
        loss_threshold: ALES parameter α
        patience: ALES parameter P
        min_epochs: ALES parameter E_min
    
    Returns:
        best_sample: Best reconstructed image
        best_metrics: Dict with PSNR, SSIM, LPIPS
        psnr_curve: Dict with PSNR history
    """
    
    # ══════════════════════════════════════════════════════════════════
    # Setup: Random seed and logging
    # ══════════════════════════════════════════════════════════════════
    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
    
    # Log configuration to TensorBoard
    if writer is not None and img_index is not None:
        config_str = (
            f'Algorithm: SESD\n'
            f'Iterations: {iteration}\n'
            f'Learning Rate: {lr}\n'
            f'Shortcut Step t*: {iter_step}\n'
            f'Total Steps T: {denoiser_step}\n'
            f'ALES Enabled: {use_ales}\n'
        )
        if use_ales:
            config_str += (
                f'ALES W: {window_size}\n'
                f'ALES δ_v: {var_threshold}\n'
                f'ALES α: {loss_threshold}\n'
                f'ALES P: {patience}\n'
                f'ALES E_min: {min_epochs}\n'
            )
        writer.add_text(f'sesd/Image_{img_index}/Config', config_str, 0)
        
        # Log reference and measurement images
        if ref_img.dim() == 4:
            ref_to_log = (ref_img[0] + 1) / 2
        else:
            ref_to_log = (ref_img + 1) / 2
        writer.add_image(f'sesd/Image_{img_index}/Reference', ref_to_log, 0)
        
        if y_n.dim() == 4:
            y_to_log = (y_n[0] + 1) / 2
        else:
            y_to_log = (y_n + 1) / 2
        writer.add_image(f'sesd/Image_{img_index}/Measurement', y_to_log, 0)
    
    # ══════════════════════════════════════════════════════════════════
    # Step 1: Initialize random noise Z and shortcut to t*
    # ══════════════════════════════════════════════════════════════════
    Z = torch.randn(
        (1, 3, model_config['image_size'], model_config['image_size']), 
        device=device
    )
    
    if writer is not None and img_index is not None:
        writer.add_image(f'sesd/Image_{img_index}/Initial_Noise', (Z[0] + 1) / 2, 0)
    
    # Shortcut: Forward diffusion from T down to t* (skip first denoiser_step - iter_step steps)
    sample = Z
    with torch.no_grad():
        for i, t in enumerate(tqdm(
            list(range(denoiser_step))[::-1], 
            desc="Shortcut to t*"
        )):
            time = torch.tensor(
                [t] * (1 if ref_img.dim() == 3 else ref_img.shape[0]), 
                device=device
            )
            # Stop at t* (keep last iter_step for optimization)
            if i >= denoiser_step - iter_step:
                print(f"Shortcut stopped at step {i+1} (t*={iter_step})")
                break
            
            if i == 0:
                sample, pred_start = sampler.p_sample(
                    model=model, x=Z, t=time, 
                    measurement=y_n, measurement_cond_fn=measurement_cond_fn, 
                    mask=mask
                )
            else:
                sample, pred_start = sampler.p_sample(
                    model=model, x=sample, t=time,
                    measurement=y_n, measurement_cond_fn=measurement_cond_fn,
                    mask=mask
                )
        
        if writer is not None and img_index is not None:
            writer.add_image(
                f'sesd/Image_{img_index}/After_Shortcut', 
                (sample[0] + 1) / 2, 0
            )
    
    # ══════════════════════════════════════════════════════════════════
    # Step 2: Optimization setup
    # ══════════════════════════════════════════════════════════════════
    sample = sample.detach().clone().requires_grad_(True)
    
    # Learnable balancing parameter λ (renamed from α in rebuttal)
    lambda_param = torch.tensor(0.5, requires_grad=True, device=device)
    
    optimizer = torch.optim.Adam([
        {'params': sample, 'lr': lr},
        {'params': lambda_param, 'lr': lr * 0.1}  # Smaller LR for λ
    ])
    
    # Loss functions
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    l1_loss = nn.L1Loss().to(device)
    
    # Metrics storage
    losses = []
    psnrs = []
    ssims = []
    lpipss = []
    
    # ══════════════════════════════════════════════════════════════════
    # Step 3: ALES early stopper initialization
    # ══════════════════════════════════════════════════════════════════
    if use_ales:
        early_stopper = ESWithWMV(
            window_size=window_size,
            var_threshold=var_threshold,
            alpha=loss_threshold,
            patience=patience,
            min_epochs=min_epochs,
            verbose=True
        )
    
    best_loss = float('inf')
    best_sample = None
    best_metrics = None
    best_epoch = 0
    
    # ══════════════════════════════════════════════════════════════════
    # Step 4: Optimization loop with ALES
    # ══════════════════════════════════════════════════════════════════
    pbar = tqdm(range(iteration), desc="SESD Optimization")
    for epoch in pbar:
        model.eval()
        optimizer.zero_grad()
        
        # ─────────────────────────────────────────────────────────────
        # A. Reverse diffusion from t* to 0 (denoising)
        # ─────────────────────────────────────────────────────────────
        x_t = sample
        x_t.requires_grad_(True)
        
        for i, t in enumerate(list(range(iter_step))[::-1]):
            time = torch.tensor(
                [t] * (1 if ref_img.dim() == 3 else ref_img.shape[0]),
                device=device
            )
            x_t, pred_start = sampler.p_sample(
                model=model, x=x_t, t=time,
                measurement=y_n, measurement_cond_fn=measurement_cond_fn,
                mask=mask
            )
        
        # ─────────────────────────────────────────────────────────────
        # B. Data consistency acceleration via gradient
        # ─────────────────────────────────────────────────────────────
        try:
            if mask is not None:
                difference = y_n - operator.forward(x_t, mask=mask)
            else:
                difference = y_n - operator.forward(x_t)
            
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(
                outputs=norm, inputs=x_t, retain_graph=True
            )[0]
            
            # Data consistency update v_k
            v_k = x_t - norm_grad
            
            # ─────────────────────────────────────────────────────────
            # C. Fusion with learnable λ
            # ─────────────────────────────────────────────────────────
            current_lambda = torch.sigmoid(lambda_param)
            x_k = current_lambda * x_t + (1 - current_lambda) * v_k
            
            # ─────────────────────────────────────────────────────────
            # D. Loss computation
            # ─────────────────────────────────────────────────────────
            if mask is not None:
                loss = l1_loss(operator.forward(x_k, mask=mask), y_n)
            else:
                loss = l1_loss(operator.forward(x_k), y_n)
            
            losses.append(loss.item())
            
            # ─────────────────────────────────────────────────────────
            # E. Backpropagation
            # ─────────────────────────────────────────────────────────
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # ─────────────────────────────────────────────────────────
            # F. Metrics computation
            # ─────────────────────────────────────────────────────────
            with torch.no_grad():
                # Ensure dimension consistency
                if ref_img.dim() == 4 and x_k.dim() == 3:
                    x_k_compare = x_k.unsqueeze(0)
                else:
                    x_k_compare = x_k
                
                # Convert to numpy for metrics
                x_k_np = x_k_compare.detach().cpu().squeeze().numpy()
                ref_np = ref_img.detach().cpu().squeeze().numpy()
                
                # Ensure [H, W, C] format
                if x_k_np.shape[0] == 3 and len(x_k_np.shape) == 3:
                    x_k_np = np.transpose(x_k_np, (1, 2, 0))
                if ref_np.shape[0] == 3 and len(ref_np.shape) == 3:
                    ref_np = np.transpose(ref_np, (1, 2, 0))
                
                # Normalize to [0, 1]
                x_k_np = (x_k_np + 1) / 2
                ref_np = (ref_np + 1) / 2
                
                # PSNR
                try:
                    current_psnr = peak_signal_noise_ratio(ref_np, x_k_np, data_range=1.0)
                    psnrs.append(current_psnr)
                except Exception as e:
                    print(f"PSNR error: {e}")
                    current_psnr = 0
                    psnrs.append(current_psnr)
                
                # SSIM
                try:
                    current_ssim = structural_similarity(
                        ref_np, x_k_np, channel_axis=2, data_range=1.0
                    )
                    ssims.append(current_ssim)
                except Exception as e:
                    print(f"SSIM error: {e}")
                    current_ssim = 0
                    ssims.append(current_ssim)
                
                # LPIPS
                try:
                    x_k_lpips = torch.from_numpy(x_k_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    ref_lpips = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    # LPIPS expects [-1, 1]
                    x_k_lpips = x_k_lpips * 2 - 1
                    ref_lpips = ref_lpips * 2 - 1
                    current_lpips = loss_fn_alex(x_k_lpips, ref_lpips).item()
                    lpipss.append(current_lpips)
                except Exception as e:
                    print(f"LPIPS error: {e}")
                    current_lpips = 0
                    lpipss.append(current_lpips)
                
                # ─────────────────────────────────────────────────────
                # G. TensorBoard logging
                # ─────────────────────────────────────────────────────
                if writer is not None and img_index is not None:
                    metrics_base = f'sesd_metrics/image_{img_index}'
                    
                    writer.add_scalar(f'{metrics_base}/Loss', loss.item(), epoch)
                    writer.add_scalar(f'{metrics_base}/PSNR', current_psnr, epoch)
                    writer.add_scalar(f'{metrics_base}/SSIM', current_ssim, epoch)
                    writer.add_scalar(f'{metrics_base}/LPIPS', current_lpips, epoch)
                    writer.add_scalar(f'{metrics_base}/Lambda', current_lambda.item(), epoch)
                    
                    if epoch % 10 == 0:
                        img = x_k.detach().cpu()
                        if img.dim() == 4:
                            img = img[0]
                        img_normalized = (img + 1) / 2
                        img_normalized = torch.clamp(img_normalized, 0, 1)
                        writer.add_image(
                            f'sesd/Image_{img_index}/Progress/Epoch_{epoch}',
                            img_normalized, epoch
                        )
                
                # ─────────────────────────────────────────────────────
                # H. Track best sample
                # ─────────────────────────────────────────────────────
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_sample = x_k.clone().detach()
                    best_metrics = {
                        'psnr': current_psnr,
                        'ssim': current_ssim,
                        'lpips': current_lpips
                    }
                    best_epoch = epoch
                    
                    if writer is not None and img_index is not None:
                        writer.add_text(
                            f'sesd/Image_{img_index}/Best/Info',
                            f'Epoch: {best_epoch}\n'
                            f'Loss: {best_loss:.6f}\n'
                            f'PSNR: {current_psnr:.4f}\n'
                            f'SSIM: {current_ssim:.4f}\n'
                            f'LPIPS: {current_lpips:.4f}',
                            best_epoch
                        )
                
                # ─────────────────────────────────────────────────────
                # I. ALES early stopping check
                # ─────────────────────────────────────────────────────
                if use_ales:
                    should_stop = early_stopper(epoch, x_k, loss.item())
                    if should_stop:
                        print(f"\n🛑 ALES early stopping triggered at epoch {epoch+1}")
                        if writer is not None and img_index is not None:
                            writer.add_text(
                                f'sesd/Image_{img_index}/ALES_Stop',
                                f'Early stopped at epoch {epoch+1}',
                                epoch
                            )
                        break
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'λ': f"{current_lambda.item():.4f}",
                    'PSNR': f"{current_psnr:.4f}",
                    'SSIM': f"{current_ssim:.4f}"
                })
        
        except Exception as e:
            print(f"Error in optimization: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ══════════════════════════════════════════════════════════════════
    # Step 5: Save results
    # ══════════════════════════════════════════════════════════════════
    if best_sample is None:
        best_sample = x_k
        best_metrics = {
            'psnr': psnrs[-1] if psnrs else 0,
            'ssim': ssims[-1] if ssims else 0,
            'lpips': lpipss[-1] if lpipss else 0
        }
    
    # Save images to filesystem
    try:
        os.makedirs(os.path.join(out_path, 'recon_sesd'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'input_sesd'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'label_sesd'), exist_ok=True)
        
        def save_tensor_image(tensor, path):
            img = tensor.detach().cpu()
            if img.dim() == 4:
                img = img.squeeze(0)
            img_np = img.numpy()
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            img_np = (img_np + 1) / 2
            img_np = np.clip(img_np, 0, 1)
            plt.imsave(path, img_np)
        
        save_tensor_image(best_sample, os.path.join(out_path, 'recon_sesd', fname))
        save_tensor_image(y_n, os.path.join(out_path, 'input_sesd', fname))
        save_tensor_image(ref_img, os.path.join(out_path, 'label_sesd', fname))
        
        print(f"✅ Images saved to {out_path}/[recon|input|label]_sesd/")
    except Exception as e:
        print(f"Error saving images: {e}")
    
    # Log curves to TensorBoard
    if writer is not None and img_index is not None:
        for i, loss_val in enumerate(losses):
            writer.add_scalar(f'sesd/Image_{img_index}/Curves/Loss', loss_val, i)
        for i, psnr_val in enumerate(psnrs):
            writer.add_scalar(f'sesd/Image_{img_index}/Curves/PSNR', psnr_val, i)
        for i, ssim_val in enumerate(ssims):
            writer.add_scalar(f'sesd/Image_{img_index}/Curves/SSIM', ssim_val, i)
        for i, lpips_val in enumerate(lpipss):
            writer.add_scalar(f'sesd/Image_{img_index}/Curves/LPIPS', lpips_val, i)
    
    print(f"\n{'='*60}")
    print(f"SESD finished for {fname}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best PSNR: {best_metrics['psnr']:.4f}")
    print(f"Best SSIM: {best_metrics['ssim']:.4f}")
    print(f"Best LPIPS: {best_metrics['lpips']:.4f}")
    print(f"{'='*60}\n")
    
    psnr_curve = {'psnrs': psnrs}
    return best_sample, best_metrics, psnr_curve


# Alias for backward compatibility
acce_RED_diff = sesd
SESD = sesd
