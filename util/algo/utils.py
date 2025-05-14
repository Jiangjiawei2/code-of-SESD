# util/algo/utils.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# Note: lpips is imported at module level in other files where loss_fn_alex is passed to compute_metrics.
# If compute_metrics itself needed to initialize it, it would need the import here.

def log_metrics_to_tensorboard(writer, metrics, step, img_index=None, prefix=''):
    """Logs metrics to TensorBoard.
    
    Parameters:
        writer: TensorBoard SummaryWriter object.
        metrics: Dictionary containing metric values.
        step: Training step or epoch.
        img_index: Optional image index to differentiate logs for different images.
        prefix: Prefix for metric names.
    """
    if writer is None:
        return  # If no writer is provided, return directly.
        
    # Add appropriate prefix to each metric if provided
    tag_prefix = f"{prefix}/" if prefix else ""
    
    # If image index is provided, include it in the tag
    if img_index is not None:
        for metric_name, value in metrics.items():
            if isinstance(value, list):
                # If the value is a list, take the last value
                value = value[-1] if value else 0 # Default to 0 if list is empty
            if value is not None and not (isinstance(value, float) and np.isnan(value)): # Log if not None or NaN
                writer.add_scalar(f"{tag_prefix}{metric_name}/image_{img_index}", value, step)
    else:
        # No image index, log metrics directly
        for metric_name, value in metrics.items():
            if isinstance(value, list):
                # If the value is a list, take the last value
                value = value[-1] if value else 0
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                writer.add_scalar(f"{tag_prefix}{metric_name}", value, step)

def plot_and_log_curves(writer, losses, psnrs, ssims, lpipss, out_path, img_index=None, algo_name="Algorithm"):
    """Plots and logs training curves to TensorBoard.
    
    Parameters:
        writer: TensorBoard SummaryWriter object.
        losses: List of loss values.
        psnrs: List of PSNR values.
        ssims: List of SSIM values.
        lpipss: List of LPIPS values.
        out_path: Output path to save the plot image.
        img_index: Optional image index.
        algo_name: Name of the algorithm for labeling the plot.
    """
    # Ensure the output path exists
    os.makedirs(out_path, exist_ok=True)
    
    # Avoid matplotlib import issues in non-GUI environments
    import matplotlib
    matplotlib.use('Agg')  # Use a non-interactive backend
    import matplotlib.pyplot as plt_local # Use alias to avoid conflict if plt is imported elsewhere differently
    
    # Plot training curves
    fig, axs = plt_local.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{algo_name} Training Curves' + (f' - Image {img_index}' if img_index is not None else ''), fontsize=16)
    
    # Loss Curve
    if losses: axs[0, 0].plot(losses)
    axs[0, 0].set_title('Loss Curve')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    
    # PSNR Curve
    if psnrs: axs[0, 1].plot(psnrs)
    axs[0, 1].set_title('PSNR Curve')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('PSNR (dB)')
    
    # SSIM Curve
    if ssims: axs[1, 0].plot(ssims)
    axs[1, 0].set_title('SSIM Curve')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('SSIM')
    
    # LPIPS Curve
    if lpipss: axs[1, 1].plot(lpipss)
    axs[1, 1].set_title('LPIPS Curve')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('LPIPS')
    
    plt_local.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    
    # Path for saving image-specific curves
    if img_index is not None:
        curves_path = os.path.join(out_path, f'{algo_name}_curves_image_{img_index}.png')
    else:
        curves_path = os.path.join(out_path, f'{algo_name}_curves.png')
    
    plt_local.savefig(curves_path)
    plt_local.close(fig) # Close the figure to free memory
    
    # Add the curves plot to TensorBoard
    if writer is not None:
        try:
            # Re-import for this scope if needed, or ensure plt_local.imread exists
            curves_img_np = plt_local.imread(curves_path) # Reads as H,W,C (or H,W,A)
            
            # Convert to CHW tensor for TensorBoard
            # If RGBA, take only RGB. imread might return float64 or uint8.
            if curves_img_np.shape[-1] == 4: # RGBA
                curves_img_np = curves_img_np[..., :3]
            if curves_img_np.dtype == np.uint8:
                curves_img_np = curves_img_np.astype(np.float32) / 255.0

            img_tensor = torch.from_numpy(curves_img_np).permute(2, 0, 1)  # HWC to CHW
                        
            tag_name = f'{algo_name}/Training_Curves' + (f'/image_{img_index}' if img_index is not None else '')
            writer.add_image(tag_name, img_tensor, global_step=0) # global_step=0 for summary image
        except Exception as e:
            print(f"Error adding curves image to TensorBoard: {e}")

def compute_metrics(sample, ref_img, device, loss_fn_alex, out_path=None, epoch=None, iteration=None, metrics=None):
    """Computes PSNR, SSIM, and LPIPS metrics.
    
    Parameters:
        sample: Current sampled/reconstructed image tensor [B,C,H,W] or [C,H,W]. Assumed range [-1,1].
        ref_img: Reference image tensor [B,C,H,W] or [C,H,W]. Assumed range [-1,1].
        device: Device for LPIPS calculation.
        loss_fn_alex: Pre-initialized LPIPS loss function.
        out_path: Path to save metrics CSV (optional, if None, CSV not saved).
        epoch: Current epoch for CSV logging (optional).
        iteration: Total iterations for CSV logging (optional, for CSV clearing logic).
        metrics: Dictionary to store lists of metrics. If None, initializes a new one.
                 If provided, appends to existing lists.
    
    Returns:
        dict: Updated dictionary containing lists of 'psnr', 'ssim', 'lpips'.
              If metrics was passed, it's this dict updated. Otherwise, a new one.
    """
    # Initialize metrics dictionary if not provided (for single call use)
    # If provided, it's for accumulating metrics over epochs.
    is_accumulating = metrics is not None
    if not is_accumulating:
        metrics_output = {'psnr': [], 'ssim': [], 'lpips': []}
    else:
        metrics_output = metrics

    # Ensure sample and reference images have a batch dimension for consistency
    sample_bchw = sample.unsqueeze(0) if sample.dim() == 3 else sample
    ref_img_bchw = ref_img.unsqueeze(0) if ref_img.dim() == 3 else ref_img
    
    # Convert to numpy for skimage metrics, assuming B=1 after unsqueeze if needed.
    # Squeeze batch dim, keep C,H,W, then convert to H,W,C or H,W.
    # Assume input tensors 'sample' and 'ref_img' are in [-1,1] range.
    
    ref_numpy_chw = ref_img_bchw.squeeze(0).cpu().detach().numpy()
    output_numpy_chw = sample_bchw.squeeze(0).cpu().detach().numpy()

    channel_axis_ssim = None
    if output_numpy_chw.shape[0] == 3: # Color [C,H,W] -> [H,W,C]
        output_numpy_eval = np.transpose(output_numpy_chw, (1, 2, 0))
        ref_numpy_eval = np.transpose(ref_numpy_chw, (1, 2, 0))
        channel_axis_ssim = 2
    elif output_numpy_chw.shape[0] == 1: # Grayscale [1,H,W] -> [H,W]
        output_numpy_eval = output_numpy_chw.squeeze(0)
        ref_numpy_eval = ref_numpy_chw.squeeze(0)
        # channel_axis_ssim remains None for 2D grayscale
    else: # Fallback for unexpected shapes
        print(f"Warning: Unexpected image shape for metrics. Sample: {output_numpy_chw.shape}, Ref: {ref_numpy_chw.shape}")
        output_numpy_eval = output_numpy_chw # Use as is, might be problematic
        ref_numpy_eval = ref_numpy_chw
        # Attempt to set channel_axis if still 3D after squeeze (e.g. some other format)
        if output_numpy_eval.ndim == 3: channel_axis_ssim = -1 


    # Normalize numpy arrays to [0,1] for PSNR/SSIM with data_range=1.0
    output_numpy_norm = np.clip((output_numpy_eval + 1.0) / 2.0, 0, 1)
    ref_numpy_norm = np.clip( (ref_numpy_eval + 1.0) / 2.0, 0, 1)

    current_psnr, current_ssim, current_lpips = float('nan'), float('nan'), float('nan')

    try:
        current_psnr = peak_signal_noise_ratio(ref_numpy_norm, output_numpy_norm, data_range=1.0)
    except Exception as e:
        print(f"Error calculating PSNR: {e}")
    metrics_output['psnr'].append(current_psnr)

    try:
        win_size = min(7, ref_numpy_norm.shape[0], ref_numpy_norm.shape[1]) # Ensure window size is not too large
        if win_size % 2 == 0: win_size -=1 # Must be odd
        if win_size >=3 : # Min window size for SSIM
            current_ssim = structural_similarity(ref_numpy_norm, output_numpy_norm, channel_axis=channel_axis_ssim, data_range=1.0, win_size=win_size)
        else:
            print("Warning: Image too small for SSIM calculation with default window size.")
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
    metrics_output['ssim'].append(current_ssim)

    try:
        if loss_fn_alex is not None:
            # LPIPS expects tensors in [-1,1] range, shape [B,C,H,W]
            # sample_bchw and ref_img_bchw are already in this format and on device (or moved by LPIPS model)
            current_lpips = loss_fn_alex(sample_bchw.to(device), ref_img_bchw.to(device)).item()
        else:
            print("LPIPS model (loss_fn_alex) not provided to compute_metrics.")
    except Exception as e:
        print(f"Error calculating LPIPS: {e}")
    metrics_output['lpips'].append(current_lpips)

    # Optional: Write current metrics to a CSV file if out_path and epoch are provided
    if out_path is not None and epoch is not None:
        os.makedirs(out_path, exist_ok=True)
        csv_file_path = os.path.join(out_path, "metrics_curve_per_image.csv") # Name implies per image
        
        # Check if file exists and is empty to write header
        # This logic creates/appends to a CSV for the duration of one image's optimization
        write_header = not (os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0) if epoch == 0 else False

        import csv # Import locally as it's optional
        with open(csv_file_path, mode="a", newline="") as file:
            csv_writer = csv.writer(file)
            if write_header:
                csv_writer.writerow(["Epoch", "PSNR", "SSIM", "LPIPS"])
            csv_writer.writerow([epoch, current_psnr, current_ssim, current_lpips])
        
        # Clear the CSV file after the last epoch of an image's optimization process
        if iteration is not None and epoch == iteration - 1:
            print(f"Clearing metrics CSV {csv_file_path} after final epoch {epoch} for current image.")
            open(csv_file_path, 'w').close() 
            
    if is_accumulating: # If called to update a passed dict
        return metrics_output
    else: # If called for a single computation, return the values directly
        return {'psnr': current_psnr, 'ssim': current_ssim, 'lpips': current_lpips}


# Early Stopping Classes
class EarlyStopping:
    """Basic early stopping strategy to stop training when a monitored metric has stopped improving."""
    def __init__(self, patience=5, min_delta=0, verbose=False, mode='min'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            mode (str): One of {"min", "max"}. In "min" mode, training will stop when the quantity
                        monitored has stopped decreasing; in "max" mode it will stop when the quantity
                        monitored has stopped increasing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

        if self.mode == 'min':
            self.val_op = np.less
            self.best_op_init = float('inf')
        elif self.mode == 'max':
            self.val_op = np.greater
            self.best_op_init = float('-inf')
        else:
            raise ValueError(f"Mode {self.mode} is unknown. Choose 'min' or 'max'.")
        self.best_score = self.best_op_init


    def __call__(self, current_score):
        """Checks if training should stop."""
        score_improved = False
        if self.best_score == self.best_op_init or \
           (self.mode == 'min' and current_score < self.best_score - self.min_delta) or \
           (self.mode == 'max' and current_score > self.best_score + self.min_delta):
            self.best_score = current_score
            self.counter = 0
            score_improved = True
            if self.verbose:
                print(f'EarlyStopping: New best score: {self.best_score:.6f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop, score_improved # Return stop flag and if score improved

    def stop_training(self):
        """Returns true if training should be stopped."""
        return self.should_stop
    
    def reset(self):
        """Resets the early stopper's state."""
        self.counter = 0
        self.best_score = self.best_op_init
        self.should_stop = False


class ESWithWMV:
    """
    Early stopping strategy based on Weighted Moving Variance (WMV) of images and loss improvement (ALES).
    This version is more conservative in its stopping decisions.
    
    Parameters:
        window_size (int): Window size for calculating moving variance.
        var_threshold (float): Variance threshold for early stopping.
        alpha (float): Relative improvement threshold for loss-based early stopping.
        patience (int): Number of epochs to wait without sufficient improvement.
        min_epochs (int): Minimum number of epochs before considering early stopping.
        verbose (bool): Whether to print detailed messages.
    """
    def __init__(self, window_size=15, var_threshold=0.0005, alpha=0.005,
                 patience=20, min_epochs=50, verbose=True):
        self.window_size = window_size
        self.var_threshold = var_threshold
        self.alpha_improvement_threshold = alpha # Renamed for clarity
        self.patience = patience
        self.min_epochs_to_run = min_epochs # Renamed for clarity
        self.verbose = verbose
        
        # Image history for WMV
        self.image_history = []
        
        # Loss history and related variables for patience-based stopping
        self.loss_history = []
        self.best_loss_val = float('inf') # Renamed for clarity
        self.best_loss_epoch = 0 # Renamed for clarity
        self.patience_counter = 0 # Renamed for clarity
        self.stop_training_flag = False # Renamed for clarity
        
    def _calculate_weight(self, idx, current_window_size):
        """Calculates weights for WMV, giving more weight to recent samples."""
        # Linear weights: 1, 2, ..., window_size
        return (idx + 1) / sum(range(1, current_window_size + 1))
    
    def _calculate_wmv(self):
        """Calculates Weighted Moving Variance (WMV). More conservative implementation."""
        if len(self.image_history) < self.window_size:
            return float('inf') # Not enough data for a full window
        
        # Get images in the current window
        recent_images_in_window = self.image_history[-self.window_size:]
        
        # Calculate mean image over the window
        mean_image_in_window = sum(recent_images_in_window) / len(recent_images_in_window)
        
        # Calculate weighted variance
        current_weighted_variance = 0.0
        
        for i, img_tensor in enumerate(recent_images_in_window):
            weight = self._calculate_weight(i, self.window_size)
            diff_from_mean = img_tensor - mean_image_in_window
            # Pixel-wise squared difference, then mean over pixels, then weighted sum
            current_weighted_variance += weight * torch.mean(diff_from_mean * diff_from_mean).item()
            
        # Optional: Heuristic to inflate variance if not enough history, making early stopping less likely
        if len(self.image_history) < self.window_size * 2: # Less than two full windows of history
            return current_weighted_variance * 2.0 
            
        return current_weighted_variance
    
    def __call__(self, epoch, current_image_tensor, current_loss_value):
        """
        Checks if training should stop based on WMV and loss improvement. Conservative implementation.
        
        Parameters:
            epoch (int): Current epoch number.
            current_image_tensor (torch.Tensor): Current generated image.
            current_loss_value (float): Current loss value.
            
        Returns:
            bool: True if training should stop, False otherwise.
        """
        # Store current image (detached clone) and loss
        self.image_history.append(current_image_tensor.detach().clone())
        if len(self.image_history) > self.window_size * 2: # Keep history bounded to save memory
            self.image_history.pop(0)
        self.loss_history.append(current_loss_value)
        
        # Continue training if below minimum epochs
        if epoch < self.min_epochs_to_run:
            return False
        
        # 1. Check WMV based stopping (stability of generated images)
        current_wmv_value = self._calculate_wmv()
        
        wmv_trigger = False
        if current_wmv_value < self.var_threshold:
            if self.verbose:
                print(f"Epoch {epoch}: WMV ({current_wmv_value:.6f}) is below threshold ({self.var_threshold:.6f}). Checking loss stability.")
            
            # Only trigger WMV stop if loss has also stabilized recently
            if len(self.loss_history) > self.patience: # Check if enough loss history
                recent_losses_for_wmv_check = self.loss_history[-self.patience:]
                loss_variance_recent = np.var(recent_losses_for_wmv_check)
                
                # Stop if WMV is very low AND loss variance is also very low
                if current_wmv_value < self.var_threshold * 0.5 and loss_variance_recent < self.alpha_improvement_threshold * 0.01:
                    if self.verbose:
                        print(f"Epoch {epoch}: WMV is very low ({current_wmv_value:.6f}) AND loss variance is very low ({loss_variance_recent:.6f}). Triggering early stop.")
                    self.stop_training_flag = True
                    return True
                elif self.verbose :
                     print(f"Epoch {epoch}: WMV below threshold, but loss variance ({loss_variance_recent:.6f}) not low enough or WMV not 'very' low. Continuing.")


        # 2. Check loss-based patience stopping (lack of improvement)
        # Update best loss (more conservative: significant relative improvement)
        if current_loss_value < self.best_loss_val * (1.0 - self.alpha_improvement_threshold):
            self.best_loss_val = current_loss_value
            self.best_loss_epoch = epoch
            self.patience_counter = 0 # Reset patience
            if self.verbose:
                print(f"Epoch {epoch}: New best loss: {self.best_loss_val:.6f}")
        else:
            # In early phases, increase counter slower
            if epoch < self.min_epochs_to_run * 1.5: # e.g. if min_epochs=50, up to epoch 75
                self.patience_counter += 0.5 
            else:
                self.patience_counter += 1
            if self.verbose:
                 print(f"Epoch {epoch}: No significant loss improvement. Patience counter: {self.patience_counter}/{self.patience}")
        
        if self.patience_counter >= self.patience:
            # Additional check: if loss is still changing substantially, give more patience
            if len(self.loss_history) >= self.patience:
                recent_losses_for_patience = self.loss_history[-self.patience:]
                # Calculate average relative change rate in recent losses
                avg_rel_change = np.mean([abs(recent_losses_for_patience[j] - recent_losses_for_patience[j-1]) / (abs(recent_losses_for_patience[j-1]) + 1e-9)
                                         for j in range(1, len(recent_losses_for_patience))]) if len(recent_losses_for_patience) > 1 else 0
                
                # If loss still changing significantly, reset counter partially
                if avg_rel_change > self.alpha_improvement_threshold * 2: # If relative change is more than twice the improvement threshold
                    if self.verbose:
                        print(f"Epoch {epoch}: Patience met, but loss still changing significantly (avg rel change: {avg_rel_change:.6f}). Extending patience.")
                    self.patience_counter = self.patience // 2 # Give some more chances
                    return False # Do not stop yet
            
            if self.verbose:
                print(f"Epoch {epoch}: Loss-based early stopping triggered. No sufficient improvement for {self.patience} effective epochs.")
            self.stop_training_flag = True
            return True
            
        return False # Default: continue training
    
    def get_best_epoch(self):
        """Returns the epoch with the best recorded loss."""
        return self.best_loss_epoch
    
    def should_stop(self):
        """Returns whether training should be stopped."""
        return self.stop_training_flag
    
    def reset(self):
        """Resets the early stopper state for a new training run."""
        self.image_history = []
        self.loss_history = []
        self.best_loss_val = float('inf')
        self.best_loss_epoch = 0
        self.patience_counter = 0
        self.stop_training_flag = False