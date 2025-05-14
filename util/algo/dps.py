# util/algo/mpgd.py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import lpips # Keep import here for type hinting if loss_fn_alex is passed
import time # time module is imported but not used in the provided snippet
from tqdm import tqdm # tqdm is imported but not used in the provided snippet
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from util.img_utils import clear_color
from util.algo.utils import compute_metrics, plot_and_log_curves, log_metrics_to_tensorboard # Assuming these might be used elsewhere in the full file

def DPS(
    sample_fn,
    ref_img,
    y_n,
    out_path,
    fname,
    device,
    loss_fn_alex: lpips.LPIPS, # Added type hint, pass as argument
    mask=None,
    random_seed=None,
    writer=None,
    img_index=None
):
    """
    DPS (Diffusion Posterior Sampling) algorithm implementation.

    Samples, computes evaluation metrics, and saves results.
    """

    # Reset random seeds using the provided random_seed
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed) # For multi-GPU

    # Start sampling - pass writer and img_index parameters
    # .requires_grad_() removed from x_start as it's generally not needed for sampling
    x_start = torch.randn(ref_img.shape, device=device)
    sample = sample_fn(
        x_start=x_start,
        measurement=y_n,
        record=True, # Assumes sample_fn uses this
        save_root=out_path, # Assumes sample_fn uses this
        mask=mask,
        ref_img=ref_img, # Passed to sample_fn, ensure it's used if intended
        writer=writer,
        img_index=img_index
    )

    # Ensure LPIPS model is provided
    if loss_fn_alex is None:
        print("Warning: LPIPS loss function is not provided to DPS. LPIPS metric will be NaN.")
        # Optionally, create a dummy one or handle error
    
    # Save result images
    # clear_color is assumed to prepare tensors for plt.imsave (e.g., to numpy, [0,1] range)
    plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample.cpu()))
    plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n.cpu()))
    plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img.cpu()))
    
    # Convert best image and reference image to numpy format for skimage metrics
    # Assuming sample and ref_img are [B,C,H,W] and B=1.
    # If clear_color for plt.imsave already normalizes to [0,1], these NPs might be in that range.
    # Otherwise, PSNR/SSIM data_range might need adjustment.
    sample_np_chw = sample.squeeze(0).cpu().detach().numpy() # C,H,W
    ref_img_np_chw = ref_img.squeeze(0).cpu().detach().numpy() # C,H,W

    # Transpose to H,W,C for skimage if multi-channel, or use as is if grayscale (C=1)
    if sample_np_chw.shape[0] == 1: # Grayscale
        sample_np_eval = sample_np_chw[0] # H,W
        ref_img_np_eval = ref_img_np_chw[0] # H,W
        channel_axis_ssim = None # SSIM for 2D grayscale
    elif sample_np_chw.shape[0] == 3: # Color
        sample_np_eval = np.transpose(sample_np_chw, (1, 2, 0)) # H,W,C
        ref_img_np_eval = np.transpose(ref_img_np_chw, (1, 2, 0)) # H,W,C
        channel_axis_ssim = 2 # SSIM for HWC color
    else:
        # Fallback or error for unexpected channel count
        print(f"Warning: Unexpected channel count {sample_np_chw.shape[0]} for evaluation. Metrics might be incorrect.")
        sample_np_eval = np.transpose(sample_np_chw, (1, 2, 0)) 
        ref_img_np_eval = np.transpose(ref_img_np_chw, (1, 2, 0))
        channel_axis_ssim = -1 # Or appropriate axis

    # Assuming data for PSNR/SSIM is in range [-1, 1] if coming directly from model output.
    # Data range for PSNR/SSIM should match this. If it was [0,1] then data_range=1.0.
    # For [-1,1], data_range=2.0.
    # The original code used data_range=1 for SSIM, implying inputs were in a range of 1 unit (e.g. [0,1]).
    # Let's assume inputs for metrics are normalized to [0,1] for consistency with data_range=1.
    # This means tensors from diffusion models (often [-1,1]) need normalization before these metrics.
    
    # Normalize numpy arrays to [0,1] if they are in [-1,1] for PSNR/SSIM with data_range=1
    # This step is crucial if the original tensors were in [-1,1]
    sample_np_eval_norm = (sample_np_eval + 1.0) / 2.0 if np.min(sample_np_eval) < -0.1 else sample_np_eval
    ref_img_np_eval_norm = (ref_img_np_eval + 1.0) / 2.0 if np.min(ref_img_np_eval) < -0.1 else ref_img_np_eval
    sample_np_eval_norm = np.clip(sample_np_eval_norm, 0, 1)
    ref_img_np_eval_norm = np.clip(ref_img_np_eval_norm, 0, 1)

    # Calculate PSNR
    final_psnr = peak_signal_noise_ratio(ref_img_np_eval_norm, sample_np_eval_norm, data_range=1.0)
    # Calculate SSIM
    final_ssim = structural_similarity(ref_img_np_eval_norm, sample_np_eval_norm, channel_axis=channel_axis_ssim, data_range=1.0)
    
    # Calculate LPIPS
    # LPIPS expects tensors in [-1,1] range, shape [B,C,H,W]
    # 'sample' and 'ref_img' are already tensors, potentially in [-1,1] and on device.
    final_lpips = float('nan')
    if loss_fn_alex is not None:
        try:
            # Ensure tensors are in [-1,1] range before passing to LPIPS
            # If 'sample' and 'ref_img' are already in this range, use them directly.
            # Assuming 'sample' and 'ref_img' are already in [-1,1] from the diffusion model
            final_lpips = loss_fn_alex(ref_img, sample).item()
        except Exception as e:
            print(f"Error calculating LPIPS: {e}")


    # Record final metrics
    final_metric_values = {
        'psnr': final_psnr,
        'ssim': final_ssim,
        'lpips': final_lpips
    }
    
    # Log final metrics to TensorBoard
    if writer is not None and img_index is not None:
        if not np.isnan(final_psnr): writer.add_scalar(f'DPS/Final/PSNR', final_psnr, img_index)
        if not np.isnan(final_ssim): writer.add_scalar(f'DPS/Final/SSIM', final_ssim, img_index)
        if not np.isnan(final_lpips): writer.add_scalar(f'DPS/Final/LPIPS', final_lpips, img_index)
    
    print(f"DPS Final metrics for {fname}:")
    print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
    return sample, final_metric_values


def save_tensor_image(tensor, path):
    """
    Save tensor image to file - enhanced version.
    Saves the first channel as a grayscale image.
    Assumes input tensor is [B,C,H,W] or [C,H,W] in [-1,1] range.
    """
    # Ensure it's a CPU tensor and detach gradients
    img_tensor = tensor.detach().cpu()

    # Extract the first channel if multiple channels exist
    # [B, C, H, W] -> take the first channel
    if img_tensor.dim() == 4: # Batch dimension present
        img_tensor = img_tensor[0, 0, :, :] # First image in batch, first channel -> [H,W]
    elif img_tensor.dim() == 3: # No batch, C, H, W
        img_tensor = img_tensor[0, :, :]   # First channel -> [H,W]
    # If already [H,W], do nothing
    
    img_np = img_tensor.numpy()

    # Normalize to [0, 1] range (from assumed [-1,1])
    img_np_normalized = (img_np + 1.0) / 2.0
    img_np_clipped = np.clip(img_np_normalized, 0, 1) # Ensure it's within the [0,1] range

    # Save using matplotlib as grayscale image
    try:
        plt.imsave(path, img_np_clipped, cmap='gray')
    except Exception as e:
        print(f"Error saving image to {path}: {e}")
    finally:
        plt.close() # Ensure the plot is closed to free memory


def dps_mri(
    sample_fn,
    ref_img, # Expected as single-channel [B,1,H,W]
    y_n,     # Expected as single-channel [B,1,H,W]
    out_path,
    fname,
    device,
    loss_fn_alex: lpips.LPIPS, # Pass LPIPS model as argument
    mask, # Mask for the sampling process
    random_seed=None,
    writer=None,
    img_index=None
):
    """
    DPS-MRI: Diffusion Posterior Sampling adapted for MRI scenarios.
    Supports single-channel input + three-channel expanded sampling + single-channel metric calculation.
    """

    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)

    # Step 1: Expand y_n and ref_img to three channels (for sample_fn use if it expects 3 channels)
    # Diffusion models are often trained on RGB images.
    y_n_rgb = y_n.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
    ref_rgb_for_sampling = ref_img.repeat(1, 3, 1, 1) # Used if sample_fn needs 3-ch ref_img

    # Step 2: Initialize random starting point (3-channel for the sampler)
    # .requires_grad_() removed
    x_start = torch.randn_like(ref_rgb_for_sampling, device=device)

    # Step 3: Call sampling function
    sample_rgb = sample_fn( # Output is likely 3-channel
        x_start=x_start,
        measurement=y_n_rgb, # Provide 3-channel measurement
        record=True,
        save_root=out_path,
        mask=mask,
        ref_img=ref_rgb_for_sampling, # Pass 3-channel ref if sample_fn uses it
        writer=writer,
        img_index=img_index
    )
    
    # Step 4: Save images (as grayscale, using the original single-channel y_n, ref_img,
    # and the first channel of the 3-channel sample)
    save_tensor_image(sample_rgb, os.path.join(out_path, 'recon', fname)) # save_tensor_image handles 1st channel
    save_tensor_image(y_n, os.path.join(out_path, 'input', fname))
    save_tensor_image(ref_img, os.path.join(out_path, 'label', fname))

    # Step 5: Metric calculation (on original single-channel inputs/outputs)
    # Convert tensors to [H, W] numpy arrays in [0, 1] range for skimage metrics
    def to_mono_numpy_normalized(tensor_ch1): # Expects [B,1,H,W] or [1,H,W]
        img_ = tensor_ch1.squeeze().cpu().detach().numpy() # Remove B and C=1 dims -> H,W
        return np.clip((img_ + 1.0) / 2.0, 0, 1) # Normalize from [-1,1] to [0,1]

    # Use the first channel of the (potentially 3-channel) sample for metrics.
    sample_mono_for_metric = sample_rgb[:, 0:1, :, :] # Take first channel, keep dims [B,1,H,W]
    
    recon_np_norm = to_mono_numpy_normalized(sample_mono_for_metric)
    ref_np_norm = to_mono_numpy_normalized(ref_img) # Original single-channel ref_img

    final_psnr = peak_signal_noise_ratio(ref_np_norm, recon_np_norm, data_range=1.0)
    final_ssim = structural_similarity(ref_np_norm, recon_np_norm, data_range=1.0) # For 2D grayscale, no channel_axis

    # LPIPS: Models usually expect 3-channel [-1,1] input.
    # We use the original single-channel ref_img and the first channel of the sample,
    # then expand them to 3 channels for LPIPS.
    final_lpips = float('nan')
    if loss_fn_alex is not None:
        try:
            # Expand single-channel sample_mono_for_metric and ref_img to 3 channels for LPIPS
            # These are already in [-1,1] if they came from the model or were normalized for it
            sample_lpips_input = sample_mono_for_metric.repeat(1, 3, 1, 1) # [B,1,H,W] -> [B,3,H,W]
            ref_lpips_input = ref_img.repeat(1, 3, 1, 1)         # [B,1,H,W] -> [B,3,H,W]
            final_lpips = loss_fn_alex(ref_lpips_input, sample_lpips_input).item()
        except Exception as e:
            print(f"Error calculating LPIPS for MRI: {e}")

    # Step 6: Write to TensorBoard (optional)
    if writer is not None and img_index is not None:
        if not np.isnan(final_psnr): writer.add_scalar(f'DPS-MRI/Final/PSNR', final_psnr, img_index)
        if not np.isnan(final_ssim): writer.add_scalar(f'DPS-MRI/Final/SSIM', final_ssim, img_index)
        if not np.isnan(final_lpips): writer.add_scalar(f'DPS-MRI/Final/LPIPS', final_lpips, img_index)

    # Step 7: Print
    print(f"[DPS-MRI] Final metrics for {fname}:")
    print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")

    final_metric_values = {
        'psnr': final_psnr,
        'ssim': final_ssim,
        'lpips': final_lpips
    }

    # Return the (potentially 3-channel) raw sample from the sampler, and single-channel metrics
    return sample_rgb, final_metric_values