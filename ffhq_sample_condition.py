from functools import partial
import os
import argparse
import yaml
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import traceback # Import for detailed error logging

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.guided_gaussian_diffusion import create_sampler # space_timesteps removed as it was not used
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator, normalize_np, Blurkernel, generate_tilt_map
from util.logger import get_logger
from util.tools import early_stopping
import torchvision
import lpips
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import csv
import numpy as np
from util.algo import *
from motionblur.motionblur import Kernel # Assuming Kernel is used by an algo
from torch.utils.tensorboard import SummaryWriter
import json

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    parser = argparse.ArgumentParser(description="Image reconstruction using various algorithms and guided diffusion.")
    parser.add_argument('--model_config', type=str,default="./configs/model_config.yaml", help="Path to the model configuration YAML file.")
    parser.add_argument('--diffusion_config', type=str, default="./configs/mgpd_diffusion_config.yaml", help="Path to the diffusion configuration YAML file.")
    parser.add_argument('--task_config', type=str, default="./configs/super_resolution_config.yaml", help="Path to the task configuration YAML file.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to use.")
    parser.add_argument('--timestep', type=int, default=10, help="Number of timesteps for DDIM if applicable.")
    parser.add_argument('--eta', type=float, default=0, help="Eta parameter for DDIM if applicable.")
    parser.add_argument('--scale', type=float, default=17.5, help="Conditioning scale factor.")
    parser.add_argument('--method', type=str, default='mpgd_wo_proj', help="Conditioning method name (e.g., mpgd_wo_proj).") 
    parser.add_argument('--save_dir', type=str, default='./outputs/ffhq/', help="Base directory to save output images and logs.")
    parser.add_argument('--algo', type=str, default='acce_RED_diff', help="Algorithm to use for reconstruction.")
    parser.add_argument('--iter', type=int, default=200, help="Number of iterations for the chosen algorithm.")
    parser.add_argument('--lr', type=float, default=0.005, help="Learning rate for algorithms that use it.")
    parser.add_argument('--noise_scale', type=float, default=0.03, help='Scale of the noise to be added to measurements.')
    parser.add_argument('--noise_type', type=str, default='impulse', help='Type of noise to add (e.g., gaussian, impulse).')
    parser.add_argument('--iter_step', type=float, default=3, help='Step size or factor for iterative algorithms.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode, show more information.')

    args = parser.parse_args()
    
    # Logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    
    # Configure diffusion timesteps
    if args.timestep < 1000:
        diffusion_config["timestep_respacing"] = f"ddim{args.timestep}"
        diffusion_config["rescale_timesteps"] = True
    else:
        diffusion_config["timestep_respacing"] = "1000" # Ensure string if expected
        diffusion_config["rescale_timesteps"] = False
    
    diffusion_config["eta"] = args.eta

    # Configure task-specific parameters from args
    task_config["conditioning"]["method"] = args.method
    task_config["conditioning"]["params"]["scale"] = args.scale
    task_config["measurement"]["noise"]["noise_scale"] = args.noise_scale
    task_config["measurement"]["noise"]["name"] = args.noise_type

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    # Checkpoint path is hardcoded here. Consider making it an argument if it varies.
    resume_checkpoint_path = "../nonlinear/SD_style/models/ldm/celeba256/model.ckpt"
    try:
        cond_method = get_conditioning_method(cond_config['method'], operator, noiser, resume=resume_checkpoint_path, **cond_config['params'])
    except FileNotFoundError:
        logger.warning(f"Checkpoint file not found at {resume_checkpoint_path}. Will try to continue without resuming...")
        cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    
    # This is the base conditioning function; it might be specialized later (e.g., for inpainting)
    base_measurement_cond_fn = cond_method.conditioning 
    logger.info(f"Conditioning method: {task_config['conditioning']['method']}")
    
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    # The sample_fn will be fully defined before each algorithm call, potentially with a specialized measurement_cond_fn
    
    # Prepare Working directory
    # Use a more concise path, including key parameters for identification
    out_path_suffix_parts = [
        args.algo,
        args.noise_type,
        f'noise_scale{args.noise_scale}',
        f'ts{args.timestep}',
        f'eta{args.eta}',
        f'scale{args.scale}',
        f'iter{args.iter}',
        f'lr{args.lr}',
        f'iter_step{args.iter_step}'
    ]
    out_path_suffix = "_".join(out_path_suffix_parts)

    out_path = os.path.join(args.save_dir, 
                            measure_config['operator']['name'], 
                            task_config['data']['name'], 
                            task_config['conditioning']['method'],
                            out_path_suffix)
    
    # Create required directories
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Assumes images are in [0,1] and scales to [-1,1]
    ])
    try:
        dataset = get_dataset(**data_config, transforms=transform) # Make sure get_dataset is flexible enough
        loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        logger.error(traceback.format_exc())
        raise

    mask_gen_inpainting = None # Specific mask generator for inpainting
    if measure_config['operator']['name'] == 'inpainting':
        try:
            mask_gen_inpainting = mask_generator(**measure_config['mask_opt'])
        except KeyError:
            logger.warning("mask_opt configuration not found for inpainting, using default settings.")
            mask_gen_inpainting = mask_generator(
                mask_type='box', 
                mask_len_range=(model_config['image_size']//4, model_config['image_size']//2), # Example dynamic sizing
                mask_prob_range=(0.3, 0.7),
                image_size=model_config['image_size']
            )
    
    # Setup CSV file for metrics
    out_csv_path = os.path.join(out_path, 'metrics_results.csv')
    with open(out_csv_path, mode='w', newline='') as csv_file:
        csv_writer_obj = csv.writer(csv_file) # Renamed for clarity
        csv_writer_obj.writerow(['filename', 'psnr', 'ssim', 'lpips', 'execution_time']) # Added execution_time to CSV
        logger.info(f"Metrics will be saved to {out_csv_path}")

    # Store metrics for each sample
    psnrs_list = []
    ssims_list = []
    lpipss_list = []
    execution_times = []
    
    # Initialize TensorBoard SummaryWriter
    writer = None # Initialize writer to None
    try:
        tb_log_dir = os.path.join(out_path, 'tensorboard_logs')
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logging to: {tb_log_dir}")
        
        # Log experiment configuration and hyperparameters to TensorBoard
        # Combine args and key configs for comprehensive hparam logging
        hparams_dict = vars(args).copy()
        hparams_dict.update({
            'model_image_size': model_config.get('image_size'),
            'task_operator': measure_config['operator']['name'],
            'task_dataset': task_config['data']['name']
        })
        # Use an empty dict for metrics initially, will be updated later if possible
        writer.add_hparams(hparams_dict, {}) 
        
        writer.add_text('Config/Model', str(model_config))
        writer.add_text('Config/Diffusion', str(diffusion_config))
        writer.add_text('Config/Task', str(task_config))

    except Exception as e:
        logger.error(f"Error initializing TensorBoard: {e}")
        logger.error(traceback.format_exc())
        # writer remains None, subsequent checks 'if writer is not None' will handle this
    
    # Initialize LPIPS metric
    loss_fn_alex = None
    try:
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    except Exception as e:
        logger.error(f"Error initializing LPIPS: {e}. LPIPS metric will be unavailable.")
        logger.error(traceback.format_exc())
        # loss_fn_alex remains None, subsequent checks needed if LPIPS is critical
    
    all_psnr_curves = [] # To store PSNR curves for algorithms that return them
    
    #### Perform inference
    # Loop control: process a certain number of images
    # These are 0-indexed, so max_images = 10 means images 0 through 9.
    # The script assumes the dataloader returns single images, not batches of dicts
    # E.g., for i, ref_img in enumerate(loader):
    
    # min_images = 0  # Start from the 0th image in the dataset
    # max_images = 10 # Process up to (but not including) the 10th image
    # Example: To process first 5 images: min_images = 0, max_images = 5

    # Determine how many images to process.
    # If dataset is small, it might process all. Otherwise, limit to a reasonable number for testing.
    # This example processes up to 10 images.
    num_images_to_process = task_config.get("num_images_to_process", 10) # Get from config or default to 10


    for i, data_item in enumerate(loader):
        if i >= num_images_to_process:
            logger.info(f"Reached processing limit of {num_images_to_process} images.")
            break
        
        # Assuming data_item can be just the image, or (image, label)
        if isinstance(data_item, tuple) and len(data_item) == 2:
            ref_img, _ = data_item # If label is present, ignore for this script's core logic
        else:
            ref_img = data_item # Assuming it's just the image

        logger.info(f"Processing image {i+1}/{num_images_to_process} (index {i})")
        fname = f'{i:04}.png' # Padded filename for better sorting
        
        current_measurement_cond_fn = base_measurement_cond_fn # Default
        active_mask = None # Mask used for this iteration, if any
        kernel_for_algo = None # Kernel for turbulence algos

        try:
            # Ensure ref_img is on the correct device and has batch dimension
            if ref_img.dim() == 3: # C, H, W
                ref_img = ref_img.unsqueeze(0) # Add batch dimension: B, C, H, W
            ref_img = ref_img.to(device)
            
            # Log original reference image to TensorBoard
            if writer is not None:
                normalized_ref_tb = (ref_img[0].cpu() + 1) / 2  # Convert from [-1,1] to [0,1] for TB
                writer.add_image(f'Original/Image_{i}', normalized_ref_tb, i)
            
            # Prepare measurement y_n based on task
            if measure_config['operator']['name'] == 'inpainting':
                if mask_gen_inpainting is None:
                    logger.error("Inpainting task selected but mask generator is not initialized. Skipping image.")
                    continue
                try:
                    active_mask = mask_gen_inpainting(ref_img) # Generate mask
                    # Ensure mask has the correct dimensions and is on the correct device
                    if active_mask.dim() == 3: active_mask = active_mask.unsqueeze(0) # B, C, H, W or B, 1, H, W
                    if active_mask.shape[1] != ref_img.shape[1] and active_mask.shape[1] == 1:
                         active_mask = active_mask.repeat(1, ref_img.shape[1], 1, 1) # Ensure channel consistency if needed by operator
                    active_mask = active_mask.to(device)
                    
                    current_measurement_cond_fn = partial(base_measurement_cond_fn, mask=active_mask) # Specialize cond_fn
                    
                    y = operator.forward(ref_img, mask=active_mask)
                    y_n = noiser(y).to(device)
                    
                    if writer is not None: # Log mask to TensorBoard
                        writer.add_image(f'Masks/Image_{i}', active_mask[0].cpu().float(), i) # Use float for image
                except Exception as e:
                    logger.error(f"Error during inpainting setup for image {i}: {e}")
                    logger.error(traceback.format_exc())
                    continue
            
            elif measure_config['operator']['name'] == 'turbulence':
                try:
                    img_size_h, img_size_w = ref_img.shape[-2], ref_img.shape[-1]
                    # Tilt map generation (example, adjust params as needed)
                    tilt = generate_tilt_map(img_h=img_size_h, img_w=img_size_w, kernel_size=7, device=device) 
                    tilt = torch.clip(tilt, -task_config.get("tilt_clip", 2.5), task_config.get("tilt_clip", 2.5))
                    
                    # Blur kernel
                    kernel_size = task_config.get("kernel_size", 31) 
                    intensity = task_config.get("intensity", 3.0)
                    conv = Blurkernel('gaussian', kernel_size=kernel_size, device=device, std=intensity)
                    kernel_for_algo = conv.get_kernel().type(torch.float32).to(device) # B, C, K, K or 1, 1, K, K
                    if kernel_for_algo.dim() == 2: # K,K to 1,1,K,K
                         kernel_for_algo = kernel_for_algo.unsqueeze(0).unsqueeze(0)
                    if kernel_for_algo.shape[1] == 1 and ref_img.shape[1] > 1: # Grayscale kernel for color image
                         kernel_for_algo = kernel_for_algo.repeat(1,ref_img.shape[1],1,1)


                    y = operator.forward(ref_img, kernel_for_algo, tilt) # Operator needs to handle these
                    y_n = noiser(y).to(device)
                    
                    active_mask = None # No explicit mask for this type of turbulence usually

                    if writer is not None: # Log turbulence details
                        writer.add_image(f'Turbulence/Tilt_Map_{i}', (tilt[0][0].cpu().unsqueeze(0) + task_config.get("tilt_clip", 2.5))/(2*task_config.get("tilt_clip", 2.5)), i) # Normalize for viewing
                        writer.add_image(f'Turbulence/Kernel_{i}', kernel_for_algo[0,0].cpu().unsqueeze(0), i) # Visualize first channel of kernel
                except Exception as e:
                    logger.error(f"Error during turbulence setup for image {i}: {e}")
                    logger.error(traceback.format_exc())
                    continue
            
            else: # Default case (e.g., super-resolution, denoising where operator directly applies)
                active_mask = None # Or a default mask if applicable for the operator
                y = operator.forward(ref_img) # May need mask=active_mask if operator expects it
                y_n = noiser(y).to(device)
            
            # Define the specific sample_fn for this iteration
            # This ensures that if measurement_cond_fn was specialized (e.g., for inpainting), it's used.
            current_sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=current_measurement_cond_fn)

            # Set a fixed random seed for each image for reproducibility of the sampling process itself
            random_seed = 42 + i  # Use a different seed for each image
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)
                torch.cuda.manual_seed_all(random_seed) # For multi-GPU
            # For strict reproducibility, but can impact performance:
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False 
            
            # Log measurement (degraded image y_n) to TensorBoard
            if writer is not None:
                normalized_y_n_tb = (y_n[0].cpu() + 1) / 2  # Convert from [-1,1] to [0,1]
                writer.add_image(f'Measurements/Image_{i}', normalized_y_n_tb, i)
            
            # Execute chosen reconstruction algorithm
            start_time = time.time()
            sample = None
            metrics = {}
            psnr_curve_data = None # For algos returning PSNR curve

            if args.algo == 'dmplug':
                sample, metrics = DMPlug(
                    model, sampler, current_measurement_cond_fn, ref_img, y_n, args, operator, device, model_config,
                    measure_config, fname, early_stopping_threshold=1e-3, stop_patience=15, out_path=out_path,
                    iteration=args.iter, lr=args.lr, denoiser_step=args.timestep, mask=active_mask, random_seed=random_seed,
                    writer=writer, img_index=i
                )
            elif args.algo == 'dmplug_turbulence':
                sample, metrics = DMPlug_turbulence(
                    model, sampler, current_measurement_cond_fn, ref_img, y_n, args, operator, device, model_config,
                    measure_config, task_config, fname, kernel_ref=kernel_for_algo, early_stopping_threshold=1e-3, 
                    stop_patience=5, out_path=out_path, iteration=args.iter, lr=args.lr, denoiser_step=args.timestep, 
                    mask=active_mask, random_seed=random_seed, writer=writer, img_index=i
                )
            elif args.algo == 'mpgd':
                sample, metrics = mpgd(
                    current_sample_fn, ref_img, y_n, out_path, fname, device, 
                    mask=active_mask, random_seed=random_seed, writer=writer, img_index=i, loss_fn_alex=loss_fn_alex # Pass LPIPS
                )
            elif args.algo == 'acce_RED_diff':
                sample, metrics, psnr_curve_data = acce_RED_diff(
                    model, sampler, current_measurement_cond_fn, ref_img, y_n, device, model_config, measure_config, operator, fname,
                    iter_step=int(args.iter_step), iteration=args.iter, denoiser_step=args.timestep, stop_patience=15, 
                    early_stopping_threshold=0.02, lr=args.lr, out_path=out_path, mask=active_mask, random_seed=random_seed,
                    writer=writer, img_index=i, loss_fn_alex=loss_fn_alex # Pass LPIPS
                )
            elif args.algo == 'acce_RED_diff_ablation':
                 sample, metrics, psnr_curve_data = acce_RED_diff_ablation(
                    model, sampler, current_measurement_cond_fn, ref_img, y_n, device, model_config, measure_config, operator, fname,
                    iter_step=int(args.iter_step), iteration=args.iter, denoiser_step=args.timestep, stop_patience=15, 
                    early_stopping_threshold=0.02, lr=args.lr, out_path=out_path, mask=active_mask, random_seed=random_seed,
                    writer=writer, img_index=i, loss_fn_alex=loss_fn_alex # Pass LPIPS
                )
            elif args.algo == 'acce_RED_diff_turbulence':
                sample, metrics = acce_RED_diff_turbulence( # Assuming this doesn't return psnr_curve
                    model, sampler, current_measurement_cond_fn, ref_img, y_n, device, model_config, measure_config, task_config, operator, fname,
                    kernel_ref=kernel_for_algo, iter_step=int(args.iter_step), iteration=args.iter, denoiser_step=args.timestep, stop_patience=5, 
                    early_stopping_threshold=0.02, lr=args.lr, out_path=out_path, mask=active_mask, random_seed=random_seed,
                    writer=writer, img_index=i, loss_fn_alex=loss_fn_alex # Pass LPIPS
                )
            elif args.algo == 'dps':
                sample, metrics = DPS( # Assuming DPS is a defined algo in util.algo
                    current_sample_fn, ref_img, y_n, out_path, fname, device, 
                    mask=active_mask, random_seed=random_seed, writer=writer, img_index=i, loss_fn_alex=loss_fn_alex # Pass LPIPS
                )
            else:
                logger.error(f"Unknown algorithm: {args.algo}. Skipping image {i}.")
                continue
                
            end_time = time.time()
            current_execution_time = end_time - start_time
            execution_times.append(current_execution_time)
            logger.info(f"Algorithm {args.algo} execution time for image {i}: {current_execution_time:.2f} seconds.")
            
            # Log reconstruction results to TensorBoard
            if writer is not None and sample is not None:
                sample_for_tb = sample[0] if sample.dim() == 4 else sample # Assume B=1 or no B dim
                normalized_sample_tb = (sample_for_tb.cpu().detach() + 1) / 2
                writer.add_image(f'Reconstructions/Image_{i}', normalized_sample_tb, i)
                
                # Log error map
                error_map = torch.abs(ref_img[0].cpu().detach() - sample_for_tb.cpu().detach()) # Ensure on CPU
                error_map_max = error_map.max()
                if error_map_max > 1e-8 : error_map = error_map / error_map_max
                else: error_map = torch.zeros_like(error_map)
                if error_map.dim() == 2: error_map = error_map.unsqueeze(0) # Add channel dim if grayscale
                writer.add_image(f'ErrorMaps/Image_{i}', error_map, i)
            
            # Store and log metrics
            if metrics: # Ensure metrics dict is not empty
                psnr_val = metrics.get('psnr', float('nan')) # Default to NaN if not found
                ssim_val = metrics.get('ssim', float('nan'))
                lpips_val = metrics.get('lpips', float('nan'))

                psnrs_list.append(psnr_val)
                ssims_list.append(ssim_val)
                lpipss_list.append(lpips_val)
                
                # Log to CSV file, opened in 'a' mode for appending
                with open(out_csv_path, mode='a', newline='') as csv_file_append:
                    csv_writer_append = csv.writer(csv_file_append)
                    csv_writer_append.writerow([fname, psnr_val, ssim_val, lpips_val, current_execution_time])
                
                if writer is not None:
                    if not np.isnan(psnr_val): writer.add_scalar('Metrics/PSNR_per_image', psnr_val, i)
                    if not np.isnan(ssim_val): writer.add_scalar('Metrics/SSIM_per_image', ssim_val, i)
                    if not np.isnan(lpips_val): writer.add_scalar('Metrics/LPIPS_per_image', lpips_val, i)
                    writer.add_scalar('Performance/ExecutionTime_per_image', current_execution_time, i)
            
            if psnr_curve_data and 'psnrs' in psnr_curve_data: # If algorithm provides PSNR curve
                all_psnr_curves.append({
                    'image_index': i,
                    'filename': fname,
                    'psnr_curve': psnr_curve_data['psnrs'].tolist() if isinstance(psnr_curve_data['psnrs'], np.ndarray) else psnr_curve_data['psnrs']
                })
                # Save all PSNR curves to JSON after each image that produces one (can be moved outside loop for single save)
                with open(os.path.join(out_path, 'all_psnr_curves.json'), 'w') as f_json:
                    json.dump(all_psnr_curves, f_json, indent=4)

        except Exception as e:
            logger.error(f"Error processing image {i} (filename: {fname}): {e}")
            logger.error(traceback.format_exc())
            # Optionally, append NaN or error indicators to metrics lists to maintain alignment
            psnrs_list.append(float('nan'))
            ssims_list.append(float('nan'))
            lpipss_list.append(float('nan'))
            execution_times.append(float('nan'))
            continue # Continue to the next image
            
    # After processing all images, log aggregate metrics
    if writer is not None and psnrs_list: # Only if there are any results
        # Filter out NaNs before calculating mean/std for robust aggregation
        valid_psnrs = [p for p in psnrs_list if not np.isnan(p)]
        valid_ssims = [s for s in ssims_list if not np.isnan(s)]
        valid_lpipss = [l for l in lpipss_list if not np.isnan(l)]
        valid_exec_times = [t for t in execution_times if not np.isnan(t)]

        if valid_psnrs:
            avg_psnr = np.mean(valid_psnrs)
            std_psnr = np.std(valid_psnrs)
            writer.add_scalar('Metrics/Avg_PSNR', avg_psnr, num_images_to_process) # Logged at final step
            writer.add_scalar('Metrics/Std_PSNR', std_psnr, num_images_to_process)
            logger.info(f"Average PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} (over {len(valid_psnrs)} images)")

        if valid_ssims:
            avg_ssim = np.mean(valid_ssims)
            std_ssim = np.std(valid_ssims)
            writer.add_scalar('Metrics/Avg_SSIM', avg_ssim, num_images_to_process)
            writer.add_scalar('Metrics/Std_SSIM', std_ssim, num_images_to_process)
            logger.info(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f} (over {len(valid_ssims)} images)")

        if valid_lpipss:
            avg_lpips = np.mean(valid_lpipss)
            std_lpips = np.std(valid_lpipss)
            writer.add_scalar('Metrics/Avg_LPIPS', avg_lpips, num_images_to_process)
            writer.add_scalar('Metrics/Std_LPIPS', std_lpips, num_images_to_process)
            logger.info(f"Average LPIPS: {avg_lpips:.4f} ± {std_lpips:.4f} (over {len(valid_lpipss)} images)")
        
        if valid_exec_times:
            avg_execution_time = np.mean(valid_exec_times)
            total_execution_time = np.sum(valid_exec_times)
            writer.add_scalar('Performance/Avg_Execution_Time_Overall', avg_execution_time, num_images_to_process)
            writer.add_scalar('Performance/Total_Execution_Time', total_execution_time, num_images_to_process)
            logger.info(f"Average execution time per image: {avg_execution_time:.2f} seconds")
            logger.info(f"Total execution time for {len(valid_exec_times)} images: {total_execution_time:.2f} seconds")

        # Generate and save distribution plots
        try:
            # PSNR, SSIM, LPIPS distributions
            fig_dist, axes_dist = plt.subplots(1, 3, figsize=(18, 5))
            if valid_psnrs: axes_dist[0].boxplot(valid_psnrs, vert=False)
            axes_dist[0].set_yticklabels(['PSNR']) # Or set title if vert=True
            axes_dist[0].set_title('PSNR Distribution')
            if valid_ssims: axes_dist[1].boxplot(valid_ssims, vert=False)
            axes_dist[1].set_yticklabels(['SSIM'])
            axes_dist[1].set_title('SSIM Distribution')
            if valid_lpipss: axes_dist[2].boxplot(valid_lpipss, vert=False)
            axes_dist[2].set_yticklabels(['LPIPS'])
            axes_dist[2].set_title('LPIPS Distribution')
            plt.tight_layout()
            dist_path = os.path.join(out_path, 'metric_distributions.png')
            plt.savefig(dist_path)
            img_tensor = torchvision.transforms.ToTensor()(plt.imread(dist_path)) # Read back saved image
            writer.add_image('Visualizations/Metric_Distributions_Boxplot', img_tensor, 0)
            plt.close(fig_dist)

            # Bar plots for metrics per image (if lists are not too long)
            if len(valid_psnrs) <= 50: # Only plot if few images for clarity
                fig_bar, axes_bar = plt.subplots(3, 1, figsize=(15, 12)) # Changed to 3,1 for better layout
                img_indices = list(range(len(valid_psnrs))) # Assuming psnrs, ssims, lpipss have same valid length
                if valid_psnrs: axes_bar[0].bar(img_indices, valid_psnrs)
                axes_bar[0].set_title('PSNR per Image')
                axes_bar[0].set_ylabel('PSNR (dB)')
                if valid_ssims: axes_bar[1].bar(img_indices, valid_ssims)
                axes_bar[1].set_title('SSIM per Image')
                axes_bar[1].set_ylabel('SSIM')
                if valid_lpipss: axes_bar[2].bar(img_indices, valid_lpipss)
                axes_bar[2].set_title('LPIPS per Image')
                axes_bar[2].set_ylabel('LPIPS')
                axes_bar[2].set_xlabel('Processed Image Index (valid samples)')
                plt.tight_layout()
                metrics_bar_path = os.path.join(out_path, 'metrics_per_image_bar.png')
                plt.savefig(metrics_bar_path)
                img_tensor_bar = torchvision.transforms.ToTensor()(plt.imread(metrics_bar_path))
                writer.add_image('Visualizations/Metrics_per_Image_Bar', img_tensor_bar, 0)
                plt.close(fig_bar)
        except Exception as e:
            logger.error(f"Error generating metric visualization plots: {e}")
            logger.error(traceback.format_exc())
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        logger.info("TensorBoard writer closed.")
    
    logger.info(f"All processing finished. Outputs and logs are in: {out_path}")
    logger.info(f"To view TensorBoard logs, run: tensorboard --logdir=\"{tb_log_dir if 'tb_log_dir' in locals() else os.path.join(out_path, 'tensorboard_logs')}\"")

if __name__ == '__main__':
    main()