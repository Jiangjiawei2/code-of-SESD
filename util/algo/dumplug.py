# util/algo/dmplug.py
import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import lpips # For type hinting and if used directly
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from util.img_utils import clear_color
from util.algo.utils import compute_metrics, plot_and_log_curves, log_metrics_to_tensorboard

def DMPlug(
    model,
    sampler,
    measurement_cond_fn,
    ref_img,          # Reference image tensor
    y_n,              # Measurement tensor
    args,             # Arguments from argparse
    operator,         # Forward operator
    device,
    model_config,     # Model configuration dict
    measure_config,   # Measurement configuration dict
    fname,            # Filename for saving
    loss_fn_alex: lpips.LPIPS, # Pass LPIPS model as argument
    early_stopping_threshold=0.01,
    stop_patience=5,
    out_path="outputs", # Base output path
    iteration=2000,   # Number of optimization iterations
    lr=0.02,          # Learning rate for Z
    denoiser_step=3,  # Number of denoising steps per iteration
    mask=None,        # Optional mask for masked operations
    random_seed=None,
    writer=None,      # TensorBoard SummaryWriter
    img_index=None    # Index of the current image for logging
):
    """
    DMPlug algorithm: Uses a diffusion model as a prior to reconstruct images through iterative optimization.
    """
    # Set random seeds
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
    
    # Log hyperparameters and initial state to TensorBoard
    if writer is not None and img_index is not None:
        writer.add_text(f'DMPlug/Image_{img_index}/Config',
                        (f'Iterations: {iteration}\n'
                         f'Learning Rate: {lr}\n'
                         f'Denoiser Steps: {denoiser_step}\n'
                         f'Early Stopping: threshold={early_stopping_threshold}, patience={stop_patience}\n'
                         f'Random Seed: {random_seed}'), 0)
        
        # Log reference image and measurement image (normalize to [0,1] for viewing)
        writer.add_image(f'DMPlug/Image_{img_index}/Reference', (ref_img[0].cpu() + 1)/2, 0)
        writer.add_image(f'DMPlug/Image_{img_index}/Measurement', (y_n[0].cpu() + 1)/2, 0)
    
    # Initialize variables
    # Z is the latent variable we are optimizing
    Z = torch.randn((1, ref_img.shape[1], model_config['image_size'], model_config['image_size']), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([{'params': Z, 'lr': lr}])
    # loss_fn_alex is now passed as an argument
    
    criterion = torch.nn.MSELoss().to(device)
    # l1_loss = torch.nn.L1Loss() # l1_loss initialized but not used in this function

    # Log initial random noise (normalize to [0,1] for viewing)
    if writer is not None and img_index is not None:
        writer.add_image(f'DMPlug/Image_{img_index}/Initial_Noise', (Z[0].cpu().detach() + 1)/2, 0)

    # For recording metrics and optimization process
    losses_log = [] # Renamed to avoid conflict if 'losses' is used elsewhere
    psnrs_log = []
    ssims_log = []
    lpipss_log = []
    mean_changes = []  # Record mean changes for early stopping
    best_psnr = 0.0
    best_img = None
    best_epoch = 0
    
    # Main optimization loop
    for epoch in tqdm(range(iteration), desc=f"DMPlug Opt. Img {img_index or ''}"):
        model.eval() # Ensure model is in evaluation mode
        optimizer.zero_grad()
        
        # Apply the diffusion model's denoising process
        # Start with current Z, apply a few denoising steps
        current_sample = Z 
        for i in range(denoiser_step -1, -1, -1): # Iterate t from denoiser_step-1 down to 0
            t_val = torch.tensor([i] * Z.shape[0], device=device) # Z.shape[0] is batch size (1)
            # p_sample denoises current_sample based on t_val and measurement y_n
            current_sample, _ = sampler.p_sample( # pred_start is often returned but not used here
                model=model, 
                x=current_sample, 
                t=t_val, 
                measurement=y_n,
                measurement_cond_fn=measurement_cond_fn, 
                mask=mask
            )
        
        # The result of the denoising steps is our 'sample' for this epoch
        sample_from_denoiser = current_sample

        # Calculate loss based on the denoised sample
        if mask is not None:
            # Ensure mask is on the same device and has compatible dimensions
            mask_dev = mask.to(device) if mask.device != device else mask
            loss = criterion(operator.forward(sample_from_denoiser, mask=mask_dev), y_n)
        else:
            loss = criterion(operator.forward(sample_from_denoiser), y_n)
        
        # Backpropagation and optimization (updates Z)
        loss.backward() # `retain_graph=True` was present, remove if not strictly needed for this flow.
                        # If Z is the only thing gradients flow to, it might not be.
                        # However, keeping it if the original structure relied on it for complex ops.
                        # For simple Z optimization, usually not needed unless some part of sampler needs it.
                        # Assuming standard Z optimization, `retain_graph=False` (default) is fine.
        optimizer.step()
        losses_log.append(loss.item())
        
        # Calculate evaluation metrics for the current reconstruction
        with torch.no_grad():
            # Use the 'sample_from_denoiser' for metric calculation
            metrics_result = compute_metrics( # compute_metrics should handle [0,1] normalization if needed
                sample=sample_from_denoiser,
                ref_img=ref_img,
                device=device, # compute_metrics might move tensors to device if not already
                loss_fn_alex=loss_fn_alex,
                # 'out_path' and 'epoch' args for compute_metrics are removed if not used by it for this purpose
            )
            
            current_psnr = metrics_result.get('psnr', float('nan'))
            current_ssim = metrics_result.get('ssim', float('nan'))
            current_lpips = metrics_result.get('lpips', float('nan'))
            
            psnrs_log.append(current_psnr)
            ssims_log.append(current_ssim)
            lpipss_log.append(current_lpips)

            # For early stopping based on image mean stability
            current_img_np_for_mean = sample_from_denoiser.cpu().squeeze().detach().numpy()
            mean_val = np.mean(current_img_np_for_mean)
            mean_changes.append(mean_val)
            
            # Record the image with the best PSNR
            if not np.isnan(current_psnr) and current_psnr > best_psnr:
                best_psnr = current_psnr
                best_img = sample_from_denoiser.clone().detach() # Store a detached clone
                best_epoch = epoch
            
            # TensorBoard logging for current epoch
            if writer is not None and img_index is not None:
                metrics_to_log_tb = {
                    'Training/Loss': loss.item(),
                    'Metrics/PSNR': current_psnr,
                    'Metrics/SSIM': current_ssim,
                    'Metrics/LPIPS': current_lpips,
                    'Debug/ImageMean': mean_val
                }
                log_metrics_to_tensorboard(writer, metrics_to_log_tb, epoch, img_index, prefix=f'DMPlug/Image_{img_index}/Epoch')
                
                # Log intermediate process images at certain epoch intervals
                if epoch % (iteration // 10 if iteration >=10 else 1) == 0 or epoch == iteration - 1: # Log ~10 images + last
                    writer.add_image(f'DMPlug/Image_{img_index}/Intermediate/Epoch_{epoch}',
                                     (sample_from_denoiser[0].cpu().detach() + 1)/2, global_step=epoch)
                
                # Log scalar for best PSNR achieved so far, at the epoch it was achieved
                if best_img is not None and epoch == best_epoch: # Log when best is updated
                     writer.add_scalar(f'DMPlug/Image_{img_index}/BestSoFar/PSNR_at_epoch_{best_epoch}', best_psnr, global_step=epoch)

        # Early stopping check
        if epoch >= stop_patience: # Need enough entries in mean_changes
            # Check if the mean of the image has stabilized
            if len(mean_changes) > stop_patience:
                recent_changes_abs_diff = [abs(mean_changes[j] - mean_changes[j-1]) for j in range(-stop_patience + 1, 0)]
                if all(diff < early_stopping_threshold for diff in recent_changes_abs_diff):
                    print(f"DMPlug Image {img_index}: Early stopping triggered at epoch {epoch+1} due to image mean stabilization.")
                    if writer is not None and img_index is not None:
                        writer.add_text(f'DMPlug/Image_{img_index}/EarlyStopping',
                                        f'Stopped at epoch {epoch+1} (mean stability)', global_step=epoch)
                    break
    
    # If no best image was found (e.g., PSNR was always NaN or never improved), use the last sample
    if best_img is None:
        best_img = sample_from_denoiser.clone().detach() if 'sample_from_denoiser' in locals() else Z.detach()
        print(f"DMPlug Image {img_index}: Best PSNR was not updated, using last/initial image.")

    # Log training curves (PSNR, SSIM, LPIPS, Loss over epochs)
    if writer is not None and img_index is not None:
        plot_and_log_curves( # This util function should handle plotting and TB logging
            writer=writer,
            losses=losses_log,
            psnrs=psnrs_log,
            ssims=ssims_log,
            lpipss=lpipss_log,
            out_path=os.path.join(out_path, "dmplug_curves", f"img_{img_index}"), # Specific path for curves
            img_index=img_index, # For naming files/tags if needed by util
            algo_name="DMPlug",
            prefix=f"DMPlug/Image_{img_index}" # Prefix for TensorBoard tags
        )
        
    # Save final reconstructed image (the one with best PSNR)
    # Ensure 'recon' directory exists (though main script should create it)
    recon_dir = os.path.join(out_path, 'recon')
    os.makedirs(recon_dir, exist_ok=True)
    plt.imsave(os.path.join(recon_dir, fname), clear_color(best_img.cpu()))
    
    # Save input (y_n) and label (ref_img) for reference if not already done by main script
    # This might be redundant if main script already saves them per image.
    # input_dir = os.path.join(out_path, 'input')
    # label_dir = os.path.join(out_path, 'label')
    # os.makedirs(input_dir, exist_ok=True)
    # os.makedirs(label_dir, exist_ok=True)
    # plt.imsave(os.path.join(input_dir, fname), clear_color(y_n.cpu()))
    # plt.imsave(os.path.join(label_dir, fname), clear_color(ref_img.cpu()))
    
    # Calculate final metrics using the best image found
    # compute_metrics is preferred if it correctly handles normalization and device
    final_metrics_dict = compute_metrics(
        sample=best_img, 
        ref_img=ref_img, 
        device=device, 
        loss_fn_alex=loss_fn_alex
    )
    final_psnr = final_metrics_dict.get('psnr', float('nan'))
    final_ssim = final_metrics_dict.get('ssim', float('nan'))
    final_lpips = final_metrics_dict.get('lpips', float('nan'))

    # Log final metrics to TensorBoard (as a summary for this image's run)
    # Use a different step or no step for overall summary metrics per image
    if writer is not None and img_index is not None:
        writer.add_scalar(f'DMPlug/Summary/Image_{img_index}/Final_PSNR', final_psnr, 0) # Step 0 for summary
        writer.add_scalar(f'DMPlug/Summary/Image_{img_index}/Final_SSIM', final_ssim, 0)
        writer.add_scalar(f'DMPlug/Summary/Image_{img_index}/Final_LPIPS', final_lpips, 0)
        writer.add_scalar(f'DMPlug/Summary/Image_{img_index}/Best_Epoch_for_PSNR', best_epoch, 0)
    
    print(f"DMPlug Final metrics for {fname} (Best at epoch {best_epoch}):")
    print(f"PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
    return best_img, final_metrics_dict


def DMPlug_turbulence(
    model,
    sampler,
    measurement_cond_fn,
    ref_img,
    y_n,
    args, # Arguments from argparse
    operator,
    device,
    model_config,
    measure_config,
    task_config, # Task specific configurations
    fname,
    kernel_ref, # Reference kernel for turbulence
    loss_fn_alex: lpips.LPIPS, # Pass LPIPS model as argument
    early_stopping_threshold=0.01, # Threshold for statistic changes
    stop_patience=5,
    out_path="outputs",
    iteration=2000,
    lr_Z=0.02,  # Learning rate for Z
    lr_kernel=1e-1, # Learning rate for kernel
    lr_tilt=1e-7,   # Learning rate for tilt
    denoiser_step=3,
    mask=None,      # General mask, may not be used for turbulence explicitly
    random_seed=None,
    writer=None,    # TensorBoard SummaryWriter
    img_index=None  # Index of the current image for logging
):
    """
    DMPlug algorithm adapted for turbulence, optimizing image (Z), kernel, and tilt.
    """
    # Reset random seeds using the provided random_seed
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)

    # Initialize optimizable variables
    # kernel_type = task_config.get("kernel_type", "gaussian") # Get kernel type if needed
    kernel_size = task_config.get("kernel_size", 31)      # Default kernel size if not in task_config
    # intensity = task_config.get("intensity", 3.0)      # Default intensity if not in task_config

    Z = torch.randn((1, ref_img.shape[1], model_config['image_size'], model_config['image_size']), device=device, requires_grad=True)
    # Initialize trainable_kernel to be close to an identity or a small Gaussian for stability
    # A 1D representation that will be reshaped and softmaxed.
    trainable_kernel_params = torch.zeros((1, kernel_size * kernel_size), device=device) 
    # Initialize center to a higher value for a peak at the center after softmax (like a delta function)
    center_idx = (kernel_size * kernel_size) // 2
    trainable_kernel_params.data[0, center_idx] = 5.0 # Encourage initial kernel to be identity-like
    trainable_kernel_params.requires_grad = True
    
    # Initialize trainable_tilt to be small
    trainable_tilt = torch.zeros(1, 2, ref_img.shape[2], ref_img.shape[3], device=device) * 0.01 # H, W from ref_img
    trainable_tilt.requires_grad = True
    
    criterion = torch.nn.MSELoss().to(device)
    # l1_loss = torch.nn.L1Loss() # Initialized but not used in this function

    optimizer = torch.optim.Adam([
        {'params': Z, 'lr': lr_Z},
        {'params': trainable_kernel_params, 'lr': lr_kernel},
        {'params': trainable_tilt, 'lr': lr_tilt}
    ])
    
    # For logging metrics
    losses_log = []
    # Iteration-wise metrics (psnrs, ssims, lpipss) are not explicitly collected in this version's loop
    # but compute_metrics could be used if desired.
    # mean_changes, var_changes = [], [] # Record changes in mean and variance (var_changes not used)

    best_psnr = 0.0
    best_img_turbulence = None
    best_kernel_turbulence = None
    best_tilt_turbulence = None
    best_epoch_turbulence = 0

    # Optimization loop
    for epoch in tqdm(range(iteration), desc=f"DMPlug_Turbulence Opt. Img {img_index or ''}"):
        model.eval()
        optimizer.zero_grad()
        
        # Denoising Z to get a clean image estimate
        current_sample = Z
        for i in range(denoiser_step - 1, -1, -1):
            t_val = torch.tensor([i] * Z.shape[0], device=device)
            current_sample, _ = sampler.p_sample(
                model=model, 
                x=current_sample, 
                t=t_val, 
                measurement=y_n, # y_n might be used by cond_fn
                measurement_cond_fn=measurement_cond_fn, 
                mask=mask # General mask, operator for turbulence uses kernel and tilt
            )
        
        denoised_sample = torch.clamp(current_sample, -1, 1) # Clamp to valid image range

        # Form the kernel from trainable parameters
        kernel_output_softmax = F.softmax(trainable_kernel_params, dim=1) # Ensure sums to 1, positive
        current_kernel = kernel_output_softmax.view(1, 1, kernel_size, kernel_size)
        # Ensure kernel is broadcastable to image channels if operator expects it
        if denoised_sample.shape[1] > 1 and current_kernel.shape[1] == 1:
            current_kernel_to_use = current_kernel.repeat(1, denoised_sample.shape[1], 1, 1)
        else:
            current_kernel_to_use = current_kernel
            
        # Calculate reconstruction loss
        # operator.forward here is for the turbulence model
        y_reconstructed = operator.forward(denoised_sample, current_kernel_to_use, trainable_tilt)
        loss = criterion(y_reconstructed, y_n)

        # Could add regularization terms for kernel (e.g., smoothness, sparsity) or tilt if needed
        # loss += lambda_kernel_reg * torch.norm(trainable_kernel_params, p=1) 
        # loss += lambda_tilt_reg * torch.norm(trainable_tilt, p=1)

        loss.backward() # retain_graph=True might be needed if parts of graph are reused. Default is False.
        optimizer.step()
        losses_log.append(loss.item())

        # Metrics calculation and logging (less frequent for turbulence due to complexity)
        if epoch % (iteration // 10 if iteration >=10 else 1) == 0 or epoch == iteration - 1: # Log ~10 times + last
            with torch.no_grad():
                current_metrics = compute_metrics(
                    sample=denoised_sample,
                    ref_img=ref_img,
                    device=device,
                    loss_fn_alex=loss_fn_alex
                )
                current_psnr_val = current_metrics.get('psnr', float('nan'))
                
                if not np.isnan(current_psnr_val) and current_psnr_val > best_psnr:
                    best_psnr = current_psnr_val
                    best_img_turbulence = denoised_sample.clone().detach()
                    best_kernel_turbulence = current_kernel.clone().detach() # Save the 1-channel kernel
                    best_tilt_turbulence = trainable_tilt.clone().detach()
                    best_epoch_turbulence = epoch

                if writer is not None and img_index is not None:
                    writer.add_scalar(f'DMPlug_Turbulence/Image_{img_index}/Epoch/Loss', loss.item(), epoch)
                    if not np.isnan(current_psnr_val):
                         writer.add_scalar(f'DMPlug_Turbulence/Image_{img_index}/Epoch/PSNR', current_psnr_val, epoch)
                    writer.add_image(f'DMPlug_Turbulence/Image_{img_index}/Intermediate/Denoised_Sample_Epoch_{epoch}', (denoised_sample[0].cpu().detach() + 1)/2, epoch)
                    writer.add_image(f'DMPlug_Turbulence/Image_{img_index}/Intermediate/Kernel_Epoch_{epoch}', current_kernel[0].cpu().detach(), epoch) # Kernel might not be in [0,1]
                    
                    # Normalize tilt for visualization, range of tilt can vary
                    tilt_vis = trainable_tilt[0].cpu().detach()
                    tilt_min, tilt_max = tilt_vis.min(), tilt_vis.max()
                    if tilt_max > tilt_min: tilt_vis = (tilt_vis - tilt_min) / (tilt_max - tilt_min)
                    writer.add_image(f'DMPlug_Turbulence/Image_{img_index}/Intermediate/Tilt_X_Epoch_{epoch}', tilt_vis[0].unsqueeze(0), epoch)
                    writer.add_image(f'DMPlug_Turbulence/Image_{img_index}/Intermediate/Tilt_Y_Epoch_{epoch}', tilt_vis[1].unsqueeze(0), epoch)


    # Use the best found parameters
    if best_img_turbulence is None: # If no improvement or only one iteration
        best_img_turbulence = denoised_sample.clone().detach() if 'denoised_sample' in locals() else Z.detach()
        best_kernel_turbulence = current_kernel.clone().detach() if 'current_kernel' in locals() else torch.zeros_like(trainable_kernel_params.view(1,1,kernel_size,kernel_size))
        best_tilt_turbulence = trainable_tilt.clone().detach() if 'trainable_tilt' in locals() else torch.zeros_like(trainable_tilt)
        print(f"DMPlug_Turbulence Img {img_index}: Best PSNR not updated or few epochs, using last state.")


    # Save final reconstructed image, kernel, and reference kernel
    # Ensure output directories exist
    recon_dir_turb = os.path.join(out_path, 'recon_turbulence')
    label_dir_turb = os.path.join(out_path, 'label_turbulence')
    os.makedirs(recon_dir_turb, exist_ok=True)
    os.makedirs(label_dir_turb, exist_ok=True)

    plt.imsave(os.path.join(recon_dir_turb, f"img_{fname}"), clear_color(best_img_turbulence.cpu()))
    plt.imsave(os.path.join(recon_dir_turb, f"kernel_{fname}"), clear_color(best_kernel_turbulence.cpu())) # clear_color for kernel visualization
    if kernel_ref is not None:
        plt.imsave(os.path.join(label_dir_turb, f"kernel_ref_{fname}"), clear_color(kernel_ref.cpu())) # clear_color for kernel visualization

    # Final metrics calculation using the best image
    final_metrics_dict_turb = compute_metrics(
        sample=best_img_turbulence, 
        ref_img=ref_img, 
        device=device, 
        loss_fn_alex=loss_fn_alex
    )
    final_psnr_turb = final_metrics_dict_turb.get('psnr', float('nan'))
    final_ssim_turb = final_metrics_dict_turb.get('ssim', float('nan'))
    final_lpips_turb = final_metrics_dict_turb.get('lpips', float('nan'))
    
    if writer is not None and img_index is not None:
        writer.add_scalar(f'DMPlug_Turbulence/Summary/Image_{img_index}/Final_PSNR', final_psnr_turb, 0)
        writer.add_scalar(f'DMPlug_Turbulence/Summary/Image_{img_index}/Final_SSIM', final_ssim_turb, 0)
        writer.add_scalar(f'DMPlug_Turbulence/Summary/Image_{img_index}/Final_LPIPS', final_lpips_turb, 0)
        writer.add_scalar(f'DMPlug_Turbulence/Summary/Image_{img_index}/Best_Epoch_for_PSNR', best_epoch_turbulence, 0)

    print(f"DMPlug_Turbulence Final metrics for {fname} (Best at epoch {best_epoch_turbulence}):")
    print(f"PSNR: {final_psnr_turb:.4f}, SSIM: {final_ssim_turb:.4f}, LPIPS: {final_lpips_turb:.4f}")
    
    # Return the best image and its corresponding metrics
    return best_img_turbulence, final_metrics_dict_turb