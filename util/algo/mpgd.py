# util/algo/mpgd.py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import lpips # For type hinting and if LPIPS model is passed
import time
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from util.img_utils import clear_color # Assuming clear_color is defined and imported
# These are imported at module level but not used in the provided snippet directly by mpgd/mpgd_mri
# They might be used by other functions in this file, or are for other algos.
from util.algo.utils import compute_metrics, plot_and_log_curves, log_metrics_to_tensorboard
import traceback # For detailed error logging

# Helper function for saving images (generic version from original mpgd)
def _save_mpgd_image_generic(tensor_img, file_path):
    """
    Save tensor image to file. Handles multi-channel (color) or single-channel.
    Assumes tensor is [B,C,H,W] or [C,H,W], and in [-1,1] range.
    """
    if not isinstance(tensor_img, torch.Tensor):
        print(f"Warning: Expected torch.Tensor, got {type(tensor_img)} for saving to {file_path}")
        return

    img_to_save = tensor_img.detach().cpu()
    
    if img_to_save.dim() == 4:  # [B, C, H, W]
        img_to_save = img_to_save.squeeze(0)  # Convert to [C, H, W]
    
    if img_to_save.dim() != 3: # Should be [C,H,W]
        print(f"Warning: Image tensor for saving has unexpected dimensions {img_to_save.shape} for {file_path}")
        # Attempt to proceed if it's [H,W] (grayscale already squeezed)
        if img_to_save.dim() == 2: # H, W
             pass # Will be handled by cmap='gray' if no channels
        else:
            return # Cannot handle other shapes

    img_np = img_to_save.numpy()
    
    # Adjust channel order and normalize for plt.imsave
    if img_np.shape[0] == 3 and img_np.ndim == 3:  # [C, H, W] for color
        img_np = np.transpose(img_np, (1, 2, 0))  # Convert to [H, W, C]
        cmap_val = None
    elif img_np.shape[0] == 1 and img_np.ndim == 3: # [1, H, W] for grayscale
        img_np = img_np.squeeze(0) # Convert to [H,W]
        cmap_val = 'gray'
    elif img_np.ndim == 2: # Already [H,W]
        cmap_val = 'gray'
    else: # Fallback, may not display correctly
        print(f"Warning: Image numpy array has unexpected shape {img_np.shape} for {file_path}")
        cmap_val = None

    # Normalize from [-1,1] to [0,1] range for imsave
    img_np_normalized = (img_np + 1.0) / 2.0
    img_np_clipped = np.clip(img_np_normalized, 0, 1)
    
    try:
        plt.imsave(file_path, img_np_clipped, cmap=cmap_val)
    except Exception as e:
        print(f"Error in _save_mpgd_image_generic saving {file_path}: {e}")
    # plt.close() # Closing plt here might be too aggressive if used in a loop that also plots

# Helper function for saving MRI images (first channel, grayscale)
def _save_mpgd_image_mri_grayscale(tensor_img, file_path):
    """
    Save tensor's first channel as a grayscale image.
    Assumes tensor is [B,C,H,W] or [C,H,W] where C >= 1, or [H,W]. Input range [-1,1].
    """
    if not isinstance(tensor_img, torch.Tensor):
        print(f"Warning: Expected torch.Tensor, got {type(tensor_img)} for saving to {file_path}")
        return

    img_to_save = tensor_img.detach().cpu().squeeze()  # Remove batch and single channel dims if any

    if img_to_save.dim() == 3: # [C, H, W]
        img_to_save = img_to_save[0]  # Take only the first channel -> [H, W]
    elif img_to_save.dim() == 4: # Should not happen after squeeze if B=1, C=1 initially
        img_to_save = img_to_save.squeeze(0)[0] # B,C,H,W -> C,H,W -> H,W (1st chan)
    
    if img_to_save.dim() != 2: # Should be [H,W]
         print(f"Warning: MRI image tensor for saving is not 2D after processing {img_to_save.shape} for {file_path}")
         return

    img_np = img_to_save.numpy()
    
    # Normalize from [-1,1] to [0,1] range for imsave
    img_np_normalized = (img_np + 1.0) / 2.0
    img_np_clipped = np.clip(img_np_normalized, 0, 1)
    
    try:
        plt.imsave(file_path, img_np_clipped, cmap='gray')
    except Exception as e:
        print(f"Error in _save_mpgd_image_mri_grayscale saving {file_path}: {e}")
    # plt.close()


def mpgd(
    sample_fn,
    ref_img,         # Reference image tensor [B, C, H, W]
    y_n,             # Noisy measurement tensor [B, C, H, W]
    out_path,        # Output save path
    fname,           # Filename for saving
    device,
    loss_fn_alex: lpips.LPIPS, # Pass LPIPS model as argument
    mask=None,       # Optional mask tensor (for inpainting)
    random_seed=None,# Random seed for reproducibility
    writer=None,     # TensorBoard SummaryWriter object
    img_index=None   # Index of the current image, for TensorBoard logging
):
    """
    MPGD algorithm implementation: Solves linear inverse problems using a diffusion model.
    
    Parameters:
    - sample_fn: Sampling function, typically a partial function of sampler.p_sample_loop.
    - ref_img: Reference image tensor [B, C, H, W].
    - y_n: Noisy measurement tensor [B, C, H, W].
    - out_path: Output save path.
    - fname: Filename for saving.
    - device: Device to run on (CPU or GPU).
    - loss_fn_alex: Pre-initialized LPIPS model.
    - mask: Optional mask tensor (for inpainting problems).
    - random_seed: Random seed for reproducibility.
    - writer: TensorBoard SummaryWriter object.
    - img_index: Index of the current image, for TensorBoard logging.
    
    Returns:
    - sample: Reconstructed image tensor.
    - metrics: Dictionary containing PSNR, SSIM, and LPIPS.
    """
    
    # Reset random seeds using the provided random_seed
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True # For reproducibility
            torch.backends.cudnn.benchmark = False    # For reproducibility
    
    # TensorBoard logging: Log experiment configuration
    if writer is not None and img_index is not None:
        writer.add_text(f'MPGD/Image_{img_index}/Config',
                        (f'Random Seed: {random_seed}\n'
                         f'Mask Used: {"Yes" if mask is not None else "No"}'), global_step=0)
        
        # Log reference image and measurement image (normalize to [0,1] for viewing)
        writer.add_image(f'MPGD/Input/Image_{img_index}', (y_n[0].cpu().clamp(-1, 1) + 1)/2, global_step=0)
        writer.add_image(f'MPGD/Reference/Image_{img_index}', (ref_img[0].cpu().clamp(-1, 1) + 1)/2, global_step=0)
    
    # Start sampling
    sampling_start_time = time.time()
    
    # Initialize with random noise, shaped like the reference image
    x_start = torch.randn_like(ref_img, device=device)
    
    sample = None
    # Wrap sampling process with a progress bar
    # Assuming sample_fn is a single call to p_sample_loop, total=1
    with tqdm(total=1, desc=f"MPGD Sampling Img {img_index if img_index is not None else 'N/A'}") as pbar:
        try:
            # Call sampling function with all potential parameters
            sample = sample_fn(
                x_start=x_start,
                measurement=y_n,
                record=True,        # Assumes sample_fn might use this
                save_root=out_path, # Assumes sample_fn might use this
                mask=mask,
                ref_img=ref_img,    # Pass reference image
                writer=writer,      # Pass writer
                img_index=img_index # Pass image index
            )
        except TypeError as e: # Catch if sample_fn doesn't accept extra args
            print(f"Sampling function error (likely due to unsupported args): {e}. Falling back to basic call.")
            # If sampling function does not support these parameters, fall back to the basic version
            sample = sample_fn(
                x_start=x_start,
                measurement=y_n,
                record=True,
                save_root=out_path,
                mask=mask
            )
        except Exception as e:
            print(f"Generic sampling function execution error: {e}")
            traceback.print_exc()
            # Return None or raise if sampling fails critically
            return x_start, {'psnr': 0, 'ssim': 0, 'lpips': float('inf')} 
        
        pbar.update(1) # Update progress bar
    
    sampling_time = time.time() - sampling_start_time
    
    if sample is None: # If sampling failed and fallback also failed or wasn't triggered
        print(f"Error: Sampling returned None for image {img_index}.")
        sample = x_start # Use initial noise as a placeholder if all else fails
        # Or handle more gracefully, e.g., by returning error status

    # Ensure output directories exist (though main script often handles this)
    os.makedirs(os.path.join(out_path, 'recon'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'input'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'label'), exist_ok=True)
    
    # Save result images using the appropriate helper
    try:
        _save_mpgd_image_generic(sample, os.path.join(out_path, 'recon', fname))
        _save_mpgd_image_generic(y_n, os.path.join(out_path, 'input', fname))
        _save_mpgd_image_generic(ref_img, os.path.join(out_path, 'label', fname))
    except Exception as e:
        print(f"Failed to save images using primary method: {e}")
        try:
            # Fallback saving method using clear_color if _save_mpgd_image_generic fails
            print("Attempting fallback image saving method using clear_color...")
            plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample.cpu()))
            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n.cpu()))
            plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img.cpu()))
        except Exception as e2:
            print(f"Fallback saving method also failed: {e2}")
    
    # Calculate evaluation metrics
    psnr_val = float('nan')
    ssim_val = float('nan')
    lpips_val = float('nan')
    
    try:
        # Prepare numpy arrays for PSNR/SSIM, ensure [0,1] range for data_range=1.0
        # Assumes sample and ref_img are [B,C,H,W] and B=1.
        sample_np_cwh = sample.squeeze(0).cpu().detach().numpy()
        ref_np_cwh = ref_img.squeeze(0).cpu().detach().numpy()

        channel_axis_ssim = None
        if sample_np_cwh.shape[0] == 3: # Color image [C,H,W]
            sample_np_eval = np.transpose(sample_np_cwh, (1,2,0)) # -> [H,W,C]
            ref_np_eval = np.transpose(ref_np_cwh, (1,2,0))     # -> [H,W,C]
            channel_axis_ssim = 2
        elif sample_np_cwh.shape[0] == 1: # Grayscale image [1,H,W]
            sample_np_eval = sample_np_cwh.squeeze(0) # -> [H,W]
            ref_np_eval = ref_np_cwh.squeeze(0)     # -> [H,W]
        else: # Fallback for unexpected shapes
            sample_np_eval = sample_np_cwh # Use as is, might be problematic
            ref_np_eval = ref_np_cwh
            print(f"Warning: Unexpected image shape for metrics: {sample_np_eval.shape}")

        # Normalize to [0,1] for PSNR/SSIM data_range=1.0
        # This assumes original tensors were in [-1,1]
        sample_np_norm = (sample_np_eval + 1.0) / 2.0 if np.min(sample_np_eval) < -0.1 else sample_np_eval
        ref_np_norm = (ref_np_eval + 1.0) / 2.0 if np.min(ref_np_eval) < -0.1 else ref_np_eval
        sample_np_norm = np.clip(sample_np_norm, 0, 1)
        ref_np_norm = np.clip(ref_np_norm, 0, 1)

        psnr_val = peak_signal_noise_ratio(ref_np_norm, sample_np_norm, data_range=1.0)
        ssim_val = structural_similarity(ref_np_norm, sample_np_norm, channel_axis=channel_axis_ssim, data_range=1.0, win_size=min(7, ref_np_norm.shape[0], ref_np_norm.shape[1])) # Added win_size guard

        # Calculate LPIPS (expects tensors in [-1,1] range)
        if loss_fn_alex is not None:
            # 'sample' and 'ref_img' should already be [B,C,H,W] tensors in [-1,1]
            lpips_val = loss_fn_alex(sample.to(device), ref_img.to(device)).item()
        else:
            print("LPIPS model not provided, skipping LPIPS calculation.")

    except Exception as e:
        print(f"Error calculating metrics for image {img_index}: {e}")
        traceback.print_exc()

    # Store final metrics
    metrics = {
        'psnr': psnr_val,
        'ssim': ssim_val,
        'lpips': lpips_val
    }
    
    # Log final metrics to TensorBoard
    if writer is not None and img_index is not None:
        writer.add_scalar(f'MPGD/Performance/SamplingTime_sec', sampling_time, img_index)
        if not np.isnan(psnr_val): writer.add_scalar(f'MPGD/Metrics/PSNR_dB', psnr_val, img_index)
        if not np.isnan(ssim_val): writer.add_scalar(f'MPGD/Metrics/SSIM', ssim_val, img_index)
        if not np.isnan(lpips_val): writer.add_scalar(f'MPGD/Metrics/LPIPS', lpips_val, img_index)
        
        # Log reconstructed image (normalize to [0,1] for viewing)
        writer.add_image(f'MPGD/Reconstructed/Image_{img_index}', (sample[0].cpu().clamp(-1, 1) + 1)/2, global_step=img_index)
        
        try:
            # Log error map (normalize for viewing)
            error_map = torch.abs(ref_img.cpu() - sample.cpu())
            error_map_max = error_map.max()
            if error_map_max > 1e-8: error_map = error_map / error_map_max
            else: error_map = torch.zeros_like(error_map)
            writer.add_image(f'MPGD/ErrorMap/Image_{img_index}', error_map[0], global_step=img_index) # error_map[0] is [C,H,W]
        except Exception as e:
            print(f"Failed to log error map for image {img_index}: {e}")
    
    # Print final performance metrics
    print(f"MPGD Performance for Image {img_index if img_index is not None else 'N/A'}:")
    print(f"Sampling Time: {sampling_time:.4f} seconds")
    print(f"Metrics - PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}")
    
    return sample, metrics


def mpgd_mri(
    sample_fn,
    ref_img,         # Reference MRI image tensor [B, 1, H, W]
    y_n,             # Noisy MRI measurement tensor [B, 1, H, W]
    out_path,
    fname,
    device,
    loss_fn_alex: lpips.LPIPS, # Pass LPIPS model as argument
    mask=None,       # Optional mask, usually for k-space undersampling in MRI context
    random_seed=None,
    writer=None,
    img_index=None
):
    """
    MPGD-MRI: MPGD algorithm adapted for MRI reconstruction.
    Assumes single-channel (grayscale) inputs and outputs for metrics and saving.
    The sampling process itself (sample_fn) might internally handle multi-channel diffusion models.
    
    Parameters are similar to mpgd, specialized for MRI.
    ref_img and y_n are expected to be single-channel.
    """
    
    # Reset random seeds
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # TensorBoard logging
    if writer is not None and img_index is not None:
        writer.add_text(f'MPGD-MRI/Image_{img_index}/Config',
                        (f'Random Seed: {random_seed}\n'
                         f'Mask Used: {"Yes" if mask is not None else "No"}'), global_step=0)
        writer.add_image(f'MPGD-MRI/Input/Image_{img_index}', (y_n[0,0].cpu().clamp(-1,1)+1)/2, global_step=0) # Log first channel
        writer.add_image(f'MPGD-MRI/Reference/Image_{img_index}', (ref_img[0,0].cpu().clamp(-1,1)+1)/2, global_step=0) # Log first channel

    sampling_start_time = time.time()
    
    # Initialize with random noise. sample_fn might expect 3-channel noise if model is 3-channel.
    # If ref_img is [B,1,H,W], x_start will be [B,1,H,W].
    # If sample_fn requires 3-channel x_start, it needs to be adjusted or sample_fn adapted.
    # For now, assume sample_fn or its underlying model handles this.
    # Or, one might do: x_start_mono = torch.randn_like(ref_img, device=device); x_start = x_start_mono.repeat(1,3,1,1)
    x_start = torch.randn_like(ref_img, device=device) 

    sample = None
    with tqdm(total=1, desc=f"MPGD-MRI Sampling Img {img_index if img_index is not None else 'N/A'}") as pbar:
        try:
            sample = sample_fn(
                x_start=x_start, measurement=y_n, record=True, save_root=out_path, mask=mask,
                ref_img=ref_img, writer=writer, img_index=img_index
            )
        except TypeError:
            print(f"MPGD-MRI: Sampling function error (extra args). Falling back.")
            sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path, mask=mask)
        except Exception as e:
            print(f"MPGD-MRI: Generic sampling error: {e}")
            traceback.print_exc()
            return x_start, {'psnr': 0, 'ssim': 0, 'lpips': float('inf')}
        pbar.update(1)

    sampling_time = time.time() - sampling_start_time
    if sample is None:
        print(f"Error: MPGD-MRI Sampling returned None for image {img_index}.")
        sample = x_start

    # Ensure output directories
    os.makedirs(os.path.join(out_path, 'recon_mri'), exist_ok=True) # Specific dir for MRI
    os.makedirs(os.path.join(out_path, 'input_mri'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'label_mri'), exist_ok=True)

    # Save images (as grayscale)
    try:
        # sample might be 3-channel if diffusion model is RGB, _save_mpgd_image_mri_grayscale handles 1st chan.
        _save_mpgd_image_mri_grayscale(sample, os.path.join(out_path, 'recon_mri', fname))
        _save_mpgd_image_mri_grayscale(y_n, os.path.join(out_path, 'input_mri', fname))
        _save_mpgd_image_mri_grayscale(ref_img, os.path.join(out_path, 'label_mri', fname))
    except Exception as e:
        print(f"Failed to save MRI images: {e}")
        # Fallback with clear_color if needed, ensuring grayscale for MRI
        try:
            plt.imsave(os.path.join(out_path, 'recon_mri', fname), clear_color(sample.cpu())[...,0], cmap='gray') # Assuming clear_color outputs HWC, take one channel
            plt.imsave(os.path.join(out_path, 'input_mri', fname), clear_color(y_n.cpu())[...,0], cmap='gray')
            plt.imsave(os.path.join(out_path, 'label_mri', fname), clear_color(ref_img.cpu())[...,0], cmap='gray')
        except Exception as e2:
            print(f"Fallback MRI image saving also failed: {e2}")

    # Calculate metrics (for single-channel MRI)
    psnr_val, ssim_val, lpips_val = float('nan'), float('nan'), float('nan')
    try:
        # Extract first channel if sample is multi-channel, and normalize to [0,1] for PSNR/SSIM
        sample_mono_np = sample.squeeze(0)[0].cpu().detach().numpy() # [H,W] from [B,C,H,W]
        ref_mono_np = ref_img.squeeze(0)[0].cpu().detach().numpy()     # [H,W] from [B,1,H,W]

        sample_mono_norm = np.clip((sample_mono_np + 1.0) / 2.0, 0, 1)
        ref_mono_norm = np.clip((ref_mono_np + 1.0) / 2.0, 0, 1)

        psnr_val = peak_signal_noise_ratio(ref_mono_norm, sample_mono_norm, data_range=1.0)
        ssim_val = structural_similarity(ref_mono_norm, sample_mono_norm, data_range=1.0, win_size=min(7, ref_mono_norm.shape[0], ref_mono_norm.shape[1])) # Added win_size

        # LPIPS: expand single channel [0,1] numpy to 3-channel [-1,1] tensor
        if loss_fn_alex is not None:
            def to_lpips_input_tensor(np_img_01_hw): # Input is [H,W] in [0,1]
                np_img_rgb_01 = np.stack([np_img_01_hw]*3, axis=-1) # H,W,3
                tensor_n1_1 = torch.from_numpy(np_img_rgb_01).permute(2,0,1).unsqueeze(0).float().to(device) * 2.0 - 1.0
                return tensor_n1_1

            sample_lpips_in = to_lpips_input_tensor(sample_mono_norm)
            ref_lpips_in = to_lpips_input_tensor(ref_mono_norm)
            lpips_val = loss_fn_alex(ref_lpips_in, sample_lpips_in).item()
        else:
            print("LPIPS model not provided for MRI, skipping LPIPS calculation.")

    except Exception as e:
        print(f"Error calculating MRI metrics for image {img_index}: {e}")
        traceback.print_exc()

    metrics = {'psnr': psnr_val, 'ssim': ssim_val, 'lpips': lpips_val}

    # Log to TensorBoard
    if writer is not None and img_index is not None:
        writer.add_scalar(f'MPGD-MRI/Performance/SamplingTime_sec', sampling_time, img_index)
        if not np.isnan(psnr_val): writer.add_scalar(f'MPGD-MRI/Metrics/PSNR_dB', psnr_val, img_index)
        if not np.isnan(ssim_val): writer.add_scalar(f'MPGD-MRI/Metrics/SSIM', ssim_val, img_index)
        if not np.isnan(lpips_val): writer.add_scalar(f'MPGD-MRI/Metrics/LPIPS', lpips_val, img_index)
        
        # Log reconstructed image (first channel, normalized to [0,1])
        writer.add_image(f'MPGD-MRI/Reconstructed/Image_{img_index}', (sample[0,0].cpu().clamp(-1,1)+1)/2, global_step=img_index)
        
        try: # Log error map (first channel)
            error_map_mono = torch.abs(ref_img[0,0].cpu() - sample[0,0].cpu())
            error_map_max = error_map_mono.max()
            if error_map_max > 1e-8: error_map_mono = error_map_mono / error_map_max
            writer.add_image(f'MPGD-MRI/ErrorMap/Image_{img_index}', error_map_mono.unsqueeze(0), global_step=img_index) # Add channel dim
        except Exception as e:
            print(f"Failed to log MRI error map for image {img_index}: {e}")

    print(f"MPGD-MRI Performance for Image {img_index if img_index is not None else 'N/A'}:")
    print(f"Sampling Time: {sampling_time:.4f} seconds")
    print(f"Metrics - PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}")
    
    return sample, metrics # Return the raw sample (potentially 3-channel) and single-channel metrics