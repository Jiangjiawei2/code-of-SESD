# util/algo/red_diff.py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import lpips # For type hinting and if LPIPS model is passed
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from util.img_utils import clear_color # Assuming clear_color is defined
from util.algo.utils import compute_metrics as compute_metrics_util, plot_and_log_curves, log_metrics_to_tensorboard, EarlyStopping, ESWithWMV
import torch.nn as nn
import fastmri # For MRI specific functions
import traceback # For detailed error logging
from PIL import Image # For robust image saving

# --- Top-level Helper Function for Image Saving ---
def _save_algo_image(tensor_data, file_path, is_kernel=False, is_mri_grayscale=False, use_clear_color_fallback=True):
    """
    Robustly saves a tensor as an image, with fallbacks.
    Handles normalization from [-1,1] to [0,1] for standard images.
    Kernels might need different visualization.
    MRI grayscale saves the first channel.
    """
    if not isinstance(tensor_data, torch.Tensor):
        print(f"Save Error: Expected torch.Tensor, got {type(tensor_data)} for {file_path}")
        return False
    
    img_to_save = tensor_data.detach().cpu()
    saved_successfully = False

    try:
        # Default processing for plt.imsave
        if img_to_save.dim() == 4: img_to_save = img_to_save.squeeze(0) # B,C,H,W -> C,H,W
        
        if is_mri_grayscale: # Save first channel as grayscale
            if img_to_save.dim() == 3: img_to_save = img_to_save[0] # C,H,W -> H,W (1st chan)
            if img_to_save.dim() != 2:
                print(f"Save Error: MRI grayscale image not 2D after processing {img_to_save.shape} for {file_path}")
                return False # Cannot proceed
            img_np = img_to_save.numpy()
            # Assumed input tensor range is [-1,1] for images, normalized to [0,1] for saving
            img_np_normalized = np.clip((img_np + 1.0) / 2.0, 0, 1) 
            plt.imsave(file_path, img_np_normalized, cmap='gray')
        elif is_kernel: # Kernels visualization
            img_np = img_to_save.squeeze().numpy() 
            if img_np.min() < img_np.max():
                 img_np_normalized = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            else:
                 img_np_normalized = np.zeros_like(img_np)
            plt.imsave(file_path, np.clip(img_np_normalized,0,1), cmap='viridis')
        else: # Standard image (color or grayscale from C,H,W)
            img_np = img_to_save.numpy() # C,H,W
            if img_np.shape[0] == 3 and img_np.ndim == 3: # C,H,W color
                img_np = np.transpose(img_np, (1, 2, 0)) # -> H,W,C
            elif img_np.shape[0] == 1 and img_np.ndim == 3: # C,H,W grayscale
                img_np = img_np.squeeze(0) # -> H,W
            # else: assume already H,W or H,W,C if ndim==2 or (ndim==3 and shape[2]==3)
            img_np_normalized = np.clip((img_np + 1.0) / 2.0, 0, 1)
            plt.imsave(file_path, img_np_normalized, cmap='gray' if img_np_normalized.ndim==2 else None)
        
        plt.close() 
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise Exception("File not created or empty after Matplotlib save.")
        saved_successfully = True

    except Exception as e_mpl:
        print(f"Matplotlib/Primary save failed for {file_path}: {e_mpl}.")
        if use_clear_color_fallback and 'clear_color' in globals():
            print("Attempting clear_color fallback.")
            try:
                # clear_color expects a CPU tensor, might handle B,C,H,W or C,H,W
                np_array_for_imsave = clear_color(tensor_data.cpu()) 
                plt.imsave(file_path, np_array_for_imsave)
                plt.close()
                saved_successfully = True
            except Exception as e_cc:
                print(f"clear_color fallback also failed for {file_path}: {e_cc}")
        # Fallback to PIL for standard images if other methods fail
        if not saved_successfully and not is_kernel and not is_mri_grayscale:
            print("Attempting PIL fallback.")
            try:
                pil_img_tensor = tensor_data.detach().cpu()
                if pil_img_tensor.dim() == 4: pil_img_tensor = pil_img_tensor.squeeze(0)
                if pil_img_tensor.dim() == 3 and pil_img_tensor.shape[0] in [1,3]: # C,H,W
                    if pil_img_tensor.shape[0] == 1: # Grayscale
                         pil_img_tensor = pil_img_tensor.squeeze(0) # H,W
                    else: # Color
                         pil_img_tensor = pil_img_tensor.permute(1,2,0) # H,W,C
                
                img_np_pil = pil_img_tensor.numpy()
                img_np_pil_01 = np.clip((img_np_pil + 1.0) / 2.0, 0, 1)
                pil_mode = 'L' if img_np_pil_01.ndim == 2 else ('RGB' if img_np_pil_01.ndim == 3 and img_np_pil_01.shape[2] == 3 else None)
                if pil_mode:
                    img_pil_obj = Image.fromarray((img_np_pil_01 * 255).astype(np.uint8), mode=pil_mode)
                    img_pil_obj.save(file_path)
                    saved_successfully = True
                else: print("PIL fallback: unsupported image format after processing.")
            except Exception as e_pil:
                print(f"PIL save also failed for {file_path}: {e_pil}")
    
    return saved_successfully
# --- End Helper Function ---


def RED_diff(
    model,
    sampler,
    measurement_cond_fn,
    ref_img,
    y_n,
    # args, # Removed as parameter 'args' was not used in the function body
    operator,
    device,
    model_config,
    # measure_config, # Removed as not used in this function's logic
    fname,
    loss_fn_alex: lpips.LPIPS,
    out_path="outputs",
    iteration=2000,
    lr=0.02,
    denoiser_step=3,
    mask=None,
    random_seed=None,
    writer=None, 
    img_index=None 
):
    """
    RED_diff: Algorithm for solving inverse problems using a diffusion model and RED framework.
    Optimizes a latent Z and a mixing parameter alpha.
    """
    # Set random seeds
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
    
    Z_channels = ref_img.shape[1] 
    Z = torch.randn((1, Z_channels, model_config['image_size'], model_config['image_size']), device=device, requires_grad=True)
    alpha_param = torch.tensor(0.0, requires_grad=True, device=device) # Optimize pre-sigmoid alpha

    optimizer = torch.optim.Adam([{'params': Z, 'lr': lr}, {'params': alpha_param, 'lr': lr * 0.1}]) # Smaller LR for alpha
    data_fidelity_loss_fn = torch.nn.L1Loss().to(device) 

    losses_log = []
    
    # Optimization loop
    for epoch in tqdm(range(iteration), desc=f"RED_diff Opt. Img {img_index or ''}"):
        model.eval()
        optimizer.zero_grad()

        current_sample_for_denoising = Z
        for i in range(denoiser_step -1, -1, -1): 
            t_val = torch.tensor([i] * Z.shape[0], device=device)
            current_sample_for_denoising, _ = sampler.p_sample(
                model=model, x=current_sample_for_denoising, t=t_val, 
                measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask=mask
            )
        denoised_sample = current_sample_for_denoising

        denoised_sample_for_grad = denoised_sample.detach().clone().requires_grad_(True)
        op_forward_output = operator.forward(denoised_sample_for_grad, mask=mask) if mask is not None else operator.forward(denoised_sample_for_grad)
        difference = y_n - op_forward_output
        norm_of_difference = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm_of_difference, inputs=denoised_sample_for_grad, retain_graph=False)[0]
        
        v_k = denoised_sample - norm_grad 
        
        current_alpha_sig = torch.sigmoid(alpha_param) # Alpha in [0,1]
        x_k = current_alpha_sig * denoised_sample + (1 - current_alpha_sig) * v_k

        op_forward_output_xk = operator.forward(x_k, mask=mask) if mask is not None else operator.forward(x_k)
        loss = data_fidelity_loss_fn(op_forward_output_xk, y_n)
        
        loss.backward(retain_graph=True) 
        optimizer.step()
        losses_log.append(loss.item())

        if writer is not None and img_index is not None and (epoch % (iteration // 20 if iteration >=20 else 1) == 0 or epoch == iteration - 1) :
            with torch.no_grad():
                temp_metrics = compute_metrics_util(sample=x_k, ref_img=ref_img, device=device, loss_fn_alex=loss_fn_alex)
                log_metrics_to_tensorboard(writer, {
                    'Loss': loss.item(), 
                    'PSNR': temp_metrics.get('psnr'), 
                    'SSIM': temp_metrics.get('ssim'), 
                    'LPIPS': temp_metrics.get('lpips'),
                    'Alpha_Sigmoid': current_alpha_sig.item()
                    }, epoch, img_index, prefix=f'RED_diff/Image_{img_index}/Epoch')
                writer.add_image(f'RED_diff/Image_{img_index}/Intermediate/Epoch_{epoch}', (x_k[0].cpu().detach().clamp(-1,1)+1)/2, epoch)

    final_output_image = x_k.detach()
    _save_algo_image(final_output_image, os.path.join(out_path, 'recon', fname))
    _save_algo_image(y_n, os.path.join(out_path, 'input', fname))
    _save_algo_image(ref_img, os.path.join(out_path, 'label', fname))
    
    final_metrics_dict = compute_metrics_util(sample=final_output_image, ref_img=ref_img, device=device, loss_fn_alex=loss_fn_alex)
    final_psnr = final_metrics_dict.get('psnr', float('nan'))
    final_ssim = final_metrics_dict.get('ssim', float('nan'))
    final_lpips = final_metrics_dict.get('lpips', float('nan'))

    if writer is not None and img_index is not None:
        log_metrics_to_tensorboard(writer, {
            'Final_PSNR': final_psnr, 'Final_SSIM': final_ssim, 'Final_LPIPS': final_lpips, 
            'Final_Alpha_Sigmoid': torch.sigmoid(alpha_param).item()
            }, 0, img_index, prefix=f'RED_diff/Summary/Image_{img_index}')

    print(f"RED_diff Final metrics for {fname}: PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}, Alpha: {torch.sigmoid(alpha_param).item():.4f}")
    
    return final_output_image, final_metrics_dict


def acce_RED_diff_core( 
    is_mri, is_ablation,
    model, sampler, measurement_cond_fn, ref_img, y_n, device, model_config,
    operator, fname, iter_step, iteration, denoiser_step,
    lr, out_path, loss_fn_alex: lpips.LPIPS,
    k_under=None, mask_dc=None, csm=None, img_min_val=None, img_max_val=None, 
    mask_sampling=None, random_seed=None, writer=None, img_index=None
):
    """
    Core logic for Accelerated RED (Regularization by Denoising) using a diffusion model.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
    
    algo_name = "acce_RED_diff" + ("_mri" if is_mri else "") + ("_ablation" if is_ablation else "")

    if writer and img_index is not None:
        config_text = (f'Algorithm: {algo_name}\nIterations: {iteration}\nLR: {lr}\nIter Steps: {iter_step}\n'
                       f'Initial Denoiser Steps: {denoiser_step}\nSeed: {random_seed}')
        writer.add_text(f'{algo_name}/Image_{img_index}/Config', config_text, 0)
        ref_log = (ref_img.squeeze(0)[0] if ref_img.dim()==4 and ref_img.shape[1]==1 else ref_img.squeeze(0)).cpu().clamp(-1,1)
        writer.add_image(f'{algo_name}/Image_{img_index}/Reference', (ref_log + 1)/2, 0)
        y_n_log = (y_n.squeeze(0)[0] if y_n.dim()==4 and y_n.shape[1]==1 else y_n.squeeze(0)).cpu().clamp(-1,1)
        writer.add_image(f'{algo_name}/Image_{img_index}/Measurement', (y_n_log + 1)/2, 0)

    Z_channels_model = 3 
    Z = torch.randn((1, Z_channels_model, model_config['image_size'], model_config['image_size']), device=device)
    if writer and img_index is not None: writer.add_image(f'{algo_name}/Image_{img_index}/Initial_Noise_Z', (Z[0].cpu().detach().clamp(-1,1) + 1)/2, 0)

    data_fidelity_loss_fn = nn.L1Loss().to(device)
    alpha_param = torch.tensor(0.0, requires_grad=True, device=device) # Optimize pre-sigmoid alpha
    mu = torch.tensor(1.0, requires_grad=False, device=device)

    losses_log, psnrs_log, ssims_log, lpipss_log = [], [], [], []

    current_hist_sample = Z
    with torch.no_grad():
        actual_initial_den_steps = denoiser_step - iter_step
        if actual_initial_den_steps > 0 :
            for i in range(denoiser_step - 1, iter_step - 1, -1): # Corrected loop range
                t_val = torch.tensor([i] * Z.shape[0], device=device)
                current_hist_sample, _ = sampler.p_sample(model=model, x=current_hist_sample, t=t_val, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask=mask_sampling)
    initial_sample = current_hist_sample.detach().clone()
    if writer and img_index is not None: writer.add_image(f'{algo_name}/Image_{img_index}/Initial_Sample_from_Z', (initial_sample[0].cpu().detach().clamp(-1,1) + 1)/2, 0)
    
    sample_opt = initial_sample.requires_grad_(True)
    optimizer = torch.optim.Adam([{'params': sample_opt, 'lr': lr}, {'params': alpha_param, 'lr': lr * 0.1}])
    
    best_metric_val = -float('inf') # Using PSNR to track best
    best_sample_val, best_metrics_dict, best_epoch_val = None, None, 0
    
    dc_layer = DC_layer_CSM().to(device) if is_mri else None
    mri_fwd_op = mri_forward().to(device) if is_mri else None

    for epoch in tqdm(range(iteration), desc=f"{algo_name} Opt. Img {img_index or ''}"):
        model.eval()
        optimizer.zero_grad()
        
        x_t_iter = sample_opt
        for i in range(iter_step - 1, -1, -1):
            t_val = torch.tensor([i] * x_t_iter.shape[0], device=device)
            x_t_iter, _ = sampler.p_sample(model=model, x=x_t_iter, t=t_val, measurement=y_n, measurement_cond_fn=measurement_cond_fn, mask=mask_sampling)
        denoised_x_t_iter = x_t_iter

        loss_val = torch.tensor(0.0, device=device)
        x_k_output = None

        current_alpha_sig = torch.sigmoid(alpha_param) # Alpha in [0,1]

        if is_ablation:
            x_k_output = denoised_x_t_iter
            op_fwd = operator.forward(x_k_output, mask=mask_sampling) if mask_sampling else operator.forward(x_k_output)
            loss_val = data_fidelity_loss_fn(op_fwd, y_n)
        elif is_mri:
            with torch.enable_grad():
                x_t_single_mri = denoised_x_t_iter[:, 0:1, :, :]
                v_k_mri_single = dc_layer(x_t_single_mri, k_under, mask_dc, csm, img_min_val, img_max_val)
                v_k_mri = v_k_mri_single.repeat(1, Z_channels_model, 1, 1).contiguous()
                x_k_output = current_alpha_sig * denoised_x_t_iter + (1 - current_alpha_sig) * v_k_mri
                
                x_k_output_single = x_k_output[:, 0:1, :, :]
                y_reconstructed_mri = mri_fwd_op(x_k_output_single, mask_dc, csm, img_min_val, img_max_val)
                loss_val = data_fidelity_loss_fn(y_reconstructed_mri, y_n)
        else: # Standard acce_RED_diff
            with torch.enable_grad():
                denoised_x_t_iter_detached = denoised_x_t_iter.detach().clone().requires_grad_(True)
                op_fwd_detached = operator.forward(denoised_x_t_iter_detached, mask=mask_sampling) if mask_sampling else operator.forward(denoised_x_t_iter_detached)
                difference_df = y_n - op_fwd_detached
                norm_df = torch.linalg.norm(difference_df)
                df_norm_grad = torch.autograd.grad(outputs=norm_df, inputs=denoised_x_t_iter_detached, retain_graph=False)[0]
                v_k_standard = denoised_x_t_iter - mu * df_norm_grad
                x_k_output = current_alpha_sig * denoised_x_t_iter + (1 - current_alpha_sig) * v_k_standard
                op_fwd_xk = operator.forward(x_k_output, mask=mask_sampling) if mask_sampling else operator.forward(x_k_output)
                loss_val = data_fidelity_loss_fn(op_fwd_xk, y_n)

        losses_log.append(loss_val.item())
        try:
            loss_val.backward(retain_graph=True) 
            optimizer.step()
        except Exception as e_bw:
            print(f"Backward/step error {algo_name} Img {img_index} Ep {epoch}: {e_bw}"); traceback.print_exc(); break

        with torch.no_grad():
            sample_for_metrics = x_k_output[:,0:1,:,:] if is_mri else x_k_output
            curr_metrics = compute_metrics_util(sample=sample_for_metrics, ref_img=ref_img, device=device, loss_fn_alex=loss_fn_alex)
            curr_psnr = curr_metrics.get('psnr', float('nan'))
            psnrs_log.append(curr_psnr); ssims_log.append(curr_metrics.get('ssim')); lpipss_log.append(curr_metrics.get('lpips'))

            if writer and img_index is not None:
                log_metrics_to_tensorboard(writer, {'Loss': loss_val.item(), 'PSNR': curr_psnr, 'SSIM': curr_metrics.get('ssim'), 
                                                    'LPIPS': curr_metrics.get('lpips'), 'Alpha_Sigmoid': current_alpha_sig.item()},
                                           epoch, img_index, prefix=f'{algo_name}/Image_{img_index}/Epoch')
                if epoch % (iteration // 10 if iteration >=10 else 1) == 0 or epoch == iteration - 1:
                     writer.add_image(f'{algo_name}/Image_{img_index}/Intermediate/Epoch_{epoch}', (sample_for_metrics[0].cpu().detach().clamp(-1,1)+1)/2, epoch)
            
            if not np.isnan(curr_psnr) and curr_psnr > best_metric_val:
                best_metric_val = curr_psnr
                best_sample_val = x_k_output.clone().detach()
                best_metrics_dict = curr_metrics.copy()
                best_epoch_val = epoch
                if writer and img_index is not None:
                     writer.add_text(f'{algo_name}/Image_{img_index}/Best/Info', (f'Epoch: {best_epoch_val}\nPSNR: {best_metric_val:.4f}\n' +
                                     f"SSIM: {curr_metrics.get('ssim',0):.4f}\nLPIPS: {curr_metrics.get('lpips',0):.4f}\nLoss: {loss_val.item():.6f}"), best_epoch_val)
            pbar.set_postfix({'loss': loss_val.item(), 'alpha_s': current_alpha_sig.item(), 'psnr': curr_psnr})
    
    if best_sample_val is None:
        best_sample_val = x_k_output.clone().detach() if 'x_k_output' in locals() else initial_sample.detach()
        best_metrics_dict = {'psnr': psnrs_log[-1] if psnrs_log else 0,'ssim': ssims_log[-1] if ssims_log else 0, 'lpips': lpipss_log[-1] if lpipss_log else 0}
        print(f"{algo_name} Img {img_index}: Using last state as best.")

    save_subdir = algo_name.split('_')[-1] # 'mri', 'ablation', or 'diff'
    _save_algo_image(best_sample_val, os.path.join(out_path, f'recon_{save_subdir}', fname), is_mri_grayscale=is_mri)
    
    psnr_curve_dict = {'psnrs': [p for p in psnrs_log if p is not None and not np.isnan(p)]}

    if writer and img_index is not None:
        summary_prefix = f'{algo_name}/Summary/Image_{img_index}'
        if best_metrics_dict:
            log_metrics_to_tensorboard(writer, {'Final_PSNR': best_metrics_dict.get('psnr'), 'Final_SSIM': best_metrics_dict.get('ssim'),
                                     'Final_LPIPS': best_metrics_dict.get('lpips')}, 0, img_index, prefix=summary_prefix)
        writer.add_scalar(f'{summary_prefix}/Best_Epoch_for_Metric', best_epoch_val, 0)
        writer.add_scalar(f'{summary_prefix}/Final_Alpha_Sigmoid', torch.sigmoid(alpha_param).item(), 0)

    print(f"{algo_name} Final metrics for {fname} (Best at epoch {best_epoch_val}):")
    if best_metrics_dict: print(f"PSNR: {best_metrics_dict.get('psnr',0):.4f}, SSIM: {best_metrics_dict.get('ssim',0):.4f}, LPIPS: {best_metrics_dict.get('lpips',0):.4f}")
    print(f"Final Alpha (sigmoid): {torch.sigmoid(alpha_param).item():.4f}")
    
    return best_sample_val, best_metrics_dict, psnr_curve_dict

# Wrapper functions
def acce_RED_diff(model, sampler, measurement_cond_fn, ref_img, y_n, device, model_config, measure_config, operator, fname, iter_step=3, iteration=1000, denoiser_step=10, stop_patience=5, early_stopping_threshold=0.01, lr=0.02, out_path='./outputs/', mask=None, random_seed=None, writer=None, img_index=None, loss_fn_alex=None):
    if loss_fn_alex is None: loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    return acce_RED_diff_core(False, False, model, sampler, measurement_cond_fn, ref_img, y_n, device, model_config, operator, fname, iter_step, iteration, denoiser_step, lr, out_path, loss_fn_alex, mask_sampling=mask, random_seed=random_seed, writer=writer, img_index=img_index)

def acce_RED_diff_mri(model, sampler, measurement_cond_fn, ref_img, y_n, k_under, mask_dc, csm, img_min_val, img_max_val, device, model_config, measure_config, operator, fname, iter_step=3, iteration=1000, denoiser_step=10, stop_patience=5, early_stopping_threshold=0.01, lr=0.02, out_path='./outputs/', random_seed=None, writer=None, img_index=None, loss_fn_alex=None):
    if loss_fn_alex is None: loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    # Note: 'mask' from args is now mask_dc for MRI. mask_sampling for p_sample might be different or derived.
    return acce_RED_diff_core(True, False, model, sampler, measurement_cond_fn, ref_img, y_n, device, model_config, operator, fname, iter_step, iteration, denoiser_step, lr, out_path, loss_fn_alex, k_under, mask_dc, csm, img_min_val, img_max_val, mask_sampling=mask_dc, random_seed=random_seed, writer=writer, img_index=img_index)

def acce_RED_diff_ablation(model, sampler, measurement_cond_fn, ref_img, y_n, device, model_config, measure_config, operator, fname, iter_step=3, iteration=1000, denoiser_step=10, stop_patience=5, early_stopping_threshold=0.01, lr=0.02, out_path='./outputs/', mask=None, random_seed=None, writer=None, img_index=None, loss_fn_alex=None):
    if loss_fn_alex is None: loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    return acce_RED_diff_core(False, True, model, sampler, measurement_cond_fn, ref_img, y_n, device, model_config, operator, fname, iter_step, iteration, denoiser_step, lr, out_path, loss_fn_alex, mask_sampling=mask, random_seed=random_seed, writer=writer, img_index=img_index)

# --- MRI Specific Helper Functions and Modules ---
def kspace2rss(kspace_data_real): # Expects [B,Nc,H,W,2]
    """
    Computes Root-Sum-Squares (RSS) image from multi-coil k-space data.
    :param kspace_data_real: Input k-space data (batch, num_coils, H, W, 2) (real/imaginary parts).
    :return: RSS image (batch, 1, H, W).
    """
    kspace_complex = torch.view_as_complex(kspace_data_real.contiguous())
    image_space_coils = fastmri.ifft2c(kspace_complex) 
    abs_coil_images = fastmri.complex_abs(image_space_coils)
    rss_image = fastmri.rss(abs_coil_images, dim=1) # dim=1 for coils (B,Nc,H,W -> B,H,W)
    return rss_image.unsqueeze(1) # Add channel dimension: (batch, 1, H, W)

def rss_to_kspace(rss_image_normalized_b1hw, csm_maps_bnchw_real, img_min, img_max):
    """
    Converts a (normalized) RSS image back to multi-coil k-space data using CSM.
    :param rss_image_normalized_b1hw: Single-channel RSS image (batch, 1, H, W), normalized to [-1,1].
    :param csm_maps_bnchw_real: Coil Sensitivity Maps (batch, num_coils, H, W, 2).
    :param img_min: Minimum value of original RSS for de-normalization.
    :param img_max: Maximum value of original RSS for de-normalization.
    :return: Multi-coil k-space data (batch, num_coils, H, W, 2).
    """
    # De-normalize RSS image
    rss_image_physical_bhw = ((rss_image_normalized_b1hw.squeeze(1) + 1.0) / 2.0) * (img_max.view(-1,1,1) - img_min.view(-1,1,1)) + img_min.view(-1,1,1)
    
    csm_complex_bnchw = torch.view_as_complex(csm_maps_bnchw_real.contiguous())
    coil_images_complex_bnchw = csm_complex_bnchw * rss_image_physical_bhw.unsqueeze(1) # Broadcast RSS to coils
    kspace_coils_complex_bnchw = fastmri.fft2c(coil_images_complex_bnchw)
    return torch.view_as_real(kspace_coils_complex_bnchw.contiguous())

def apply_mask(kspace_data_real_bnchw2, sampling_mask_b1hw1_or_hw):
    """Applies sampling mask to k-space data (B,Nc,H,W,2). Mask is usually (B,1,H,W,1) or (H,W)."""
    kspace_complex = torch.view_as_complex(kspace_data_real_bnchw2.contiguous())
    mask_to_apply = sampling_mask_b1hw1_or_hw
    if mask_to_apply.dim() == 2: # H,W
        mask_to_apply = mask_to_apply.unsqueeze(0).unsqueeze(0).unsqueeze(-1) # -> 1,1,H,W,1
    if mask_to_apply.shape[-1] == 1: mask_to_apply = mask_to_apply.squeeze(-1) # -> B,1,H,W (or 1,1,H,W)
    
    masked_kspace_complex = kspace_complex * mask_to_apply.unsqueeze(1) # Broadcast mask over coils
    return torch.view_as_real(masked_kspace_complex.contiguous())
    
class DC_layer_CSM(nn.Module):
    def __init__(self):
        super(DC_layer_CSM, self).__init__()

    def forward(self, x_rss_norm_b1hw, k_sampled_under_bnchw2, sampling_mask_b1hw1, csm_maps_bnchw2, img_min_b, img_max_b):
        """
        Performs Data Consistency (DC) using CSMs.
        Args:
            x_rss_norm_b1hw: Input RSS image (batch, 1, H, W), normalized to [-1, 1].
            k_sampled_under_bnchw2 : Undersampled k-space data (batch, num_coils, H, W, 2).
            sampling_mask_b1hw1    : Sampling mask (batch, 1, H, W, 1).
            csm_maps_bnchw2        : Coil Sensitivity Maps (batch, num_coils, H, W, 2).
            img_min_b              : Min values of original RSS (batch) for de-normalization.
            img_max_b              : Max values of original RSS (batch) for de-normalization.
        Returns: DC corrected RSS image, re-normalized to [-1, 1] (batch, 1, H, W).
        """
        k_reconstructed_estimate_bnchw2 = rss_to_kspace(x_rss_norm_b1hw, csm_maps_bnchw2, img_min_b, img_max_b)
        
        mask_bool_b1hw = sampling_mask_b1hw1.squeeze(-1) # -> B,1,H,W
        
        k_dc_complex = ((1 - mask_bool_b1hw.float()) * torch.view_as_complex(k_reconstructed_estimate_bnchw2.contiguous()) +
                         mask_bool_b1hw.float() * torch.view_as_complex(k_sampled_under_bnchw2.contiguous()))
        k_dc_real_bnchw2 = torch.view_as_real(k_dc_complex.contiguous())

        rss_corrected_physical_b1hw = kspace2rss(k_dc_real_bnchw2)
        
        # Re-normalize. Ensure img_min_b and img_max_b are correctly shaped for broadcasting (e.g., [B,1,1,1])
        denominator = (img_max_b.view(-1,1,1,1) - img_min_b.view(-1,1,1,1) + 1e-7)
        rss_dc_normalized_b1hw = ((rss_corrected_physical_b1hw - img_min_b.view(-1,1,1,1)) / denominator) * 2.0 - 1.0
        return torch.clamp(rss_dc_normalized_b1hw, -1.0, 1.0)

class mri_forward(nn.Module): # This is A(x)
    def __init__(self):
        super(mri_forward, self).__init__()

    def forward(self, x_rss_norm_b1hw, sampling_mask_b1hw1, csm_maps_bnchw2, img_min_b, img_max_b):
        """
        MRI Forward Model: Normalized RSS Image -> K-space -> Apply Mask -> Normalized RSS Image.
        Args are same as DC_layer_CSM, without k_sampled_under.
        Returns: Undersampled RSS image, re-normalized to [-1, 1] (batch, 1, H, W).
        """
        k_reconstructed_full_bnchw2 = rss_to_kspace(x_rss_norm_b1hw, csm_maps_bnchw2, img_min_b, img_max_b)
        k_reconstructed_undersampled_bnchw2 = apply_mask(k_reconstructed_full_bnchw2, sampling_mask_b1hw1)
        rss_undersampled_physical_b1hw = kspace2rss(k_reconstructed_undersampled_bnchw2)
        
        denominator = (img_max_b.view(-1,1,1,1) - img_min_b.view(-1,1,1,1) + 1e-7)
        rss_undersampled_normalized_b1hw = ((rss_undersampled_physical_b1hw - img_min_b.view(-1,1,1,1)) / denominator) * 2.0 - 1.0
        return torch.clamp(rss_undersampled_normalized_b1hw, -1.0, 1.0)

# The local compute_metrics function from the original snippet is removed.
# It's assumed that compute_metrics_util from util.algo.utils is the standard one to be used.
# The acce_RED_diff_turbulence function from original snippet is not present in the user's last input, so it's not included here.
