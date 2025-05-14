from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.guided_gaussian_diffusion import create_sampler, space_timesteps
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
from motionblur.motionblur import Kernel
from torch.utils.tensorboard import SummaryWriter
import json
import traceback # Added for better error logging if not already present

# import os # Duplicate import, removed
                                      
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str,default="./configs/model_config.yaml")
    parser.add_argument('--diffusion_config', type=str, default="./configs/mgpd_diffusion_config.yaml")
    parser.add_argument('--task_config', type=str, default="./configs/mri_acceleration_config.yaml")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--timestep', type=int, default=10)
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--scale', type=float, default=17.5)
    parser.add_argument('--method', type=str, default='mpgd_wo_proj') # mpgd_wo_proj
    parser.add_argument('--save_dir', type=str, default='./outputs/ffhq/')
    parser.add_argument('--algo', type=str, default='acce_RED_diff_mri')  ##  dps_mri , acce_RED_diff_mri
    parser.add_argument('--iter', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--noise_scale', type=float, default=0.0, help='a value of noise_scale')
    parser.add_argument('--noise_type', type=str, default='gaussian', help='unkown noise type')
    parser.add_argument('--iter_step', type=float, default=3, help='New value for iter_step')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode, show more information')
    args = parser.parse_args()
    
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    
    if args.timestep < 1000:
        diffusion_config["timestep_respacing"] = f"ddim{args.timestep}"
        diffusion_config["rescale_timesteps"] = True
    else:
        diffusion_config["timestep_respacing"] = f"1000"
        diffusion_config["rescale_timesteps"] = False
    
    diffusion_config["eta"] = args.eta
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
    try:
        cond_method = get_conditioning_method(cond_config['method'], operator, noiser, resume = "../nonlinear/SD_style/models/ldm/celeba256/model.ckpt", **cond_config['params'])
    except FileNotFoundError:
        logger.warning("Checkpoint file not found, will try to continue...")
        cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
    
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    if args.algo == 'dps' or args.algo == 'mcg':
        sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
    else:
        # Modify here to ensure sample_fn can also receive the img_index parameter
        sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

    # Working directory
    dir_path = f"{diffusion_config['timestep_respacing']}_eta{args.eta}_scale{args.scale}"
    # Use a more concise path
    out_path = os.path.join(args.save_dir, 
                            measure_config['operator']['name'], 
                            task_config['data']['name'], 
                            args.algo, 
                            args.noise_type, 
                            f'noise_scale{args.noise_scale}', 
                            task_config['conditioning']['method'])
    
    # Create required directories
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    try:
        dataset = get_dataset(**data_config, transforms=transform)
        loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # Exception) In case of inpainting, we need to generate a mask 
    mask = None
    if measure_config['operator']['name'] == 'inpainting':
        try:
            mask_gen = mask_generator(**measure_config['mask_opt'])
        except KeyError:
            logger.warning("mask_opt configuration not found, using default configuration")
            mask_gen = mask_generator(mask_type='box', 
                                        mask_len_range=(32, 128),
                                        mask_prob_range=(0.3, 0.7),
                                        image_size=model_config['image_size'])
    
    # Set CSV file path and open file
    out_csv_path = os.path.join(out_path, 'metrics_results.csv')
    with open(out_csv_path, mode='w', newline='') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerow(['filename', 'psnr', 'ssim', 'lpips']) # Write header row

    # Store metrics for each sample
    psnrs_list = []
    ssims_list = []
    lpipss_list = []
    execution_times = []
    
    # Initialize TensorBoard SummaryWriter
    try:
        tb_log_dir = os.path.join(out_path, 'tensorboard_logs')
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logging to {tb_log_dir}")
        
        # Log experiment configuration to TensorBoard
        writer.add_text('Experiment/Algorithm', args.algo)
        writer.add_text('Experiment/Noise_Type', args.noise_type)
        writer.add_text('Experiment/Noise_Scale', str(args.noise_scale))
        writer.add_text('Experiment/Iterations', str(args.iter))
        writer.add_text('Experiment/Timesteps', str(args.timestep))
        
        # Log hyperparameters to TensorBoard
        hyperparams = {
            'algorithm': args.algo,
            'noise_type': args.noise_type,
            'noise_scale': args.noise_scale,
            'iterations': args.iter,
            'timesteps': args.timestep,
            'eta': args.eta,
            'scale': args.scale,
            'iter_step': args.iter_step,
            'method': args.method
        }
        # Use an empty dictionary for metrics, as we don't have actual results yet
        writer.add_hparams(hyperparams, {})
        
        # Add model and data configurations
        writer.add_text('Config/Model', str(model_config))
        writer.add_text('Config/Diffusion', str(diffusion_config))
        writer.add_text('Config/Task', str(task_config))
    except Exception as e:
        logger.error(f"Error initializing TensorBoard: {e}")
        writer = None # If an error occurs, set writer to None
    
    # Initialize LPIPS metric
    try:
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    except Exception as e:
        logger.error(f"Error initializing LPIPS: {e}")
        # Create a dummy loss_fn_alex to prevent subsequent code from crashing
        class DummyLPIPS:
            def __call__(self, x, y):
                return torch.tensor([0.0]).to(device)
        loss_fn_alex = DummyLPIPS()
    
    # Initialize a list outside the loop to store PSNR curves for all images
    all_psnr_curves = []
    
    #### Perform inference
    min_images = 10  # Start from the 10th image
    max_images = 20  # End at the 20th image


    for i, batch in enumerate(loader):
        ref_img = batch['original_img'].float().to(device)         # [B, 1, H, W]
        under_image_rec = batch['undersampled_img'].float().to(device) # [B, 1, H, W]
        csm = batch['csm'].to(device)                                   # [B, Nc, H, W, 2]
        mask = batch['mask'].to(device)                                 # [B, 1, W, 1]
        filenames = batch['filename']
        k_under =  batch['under_kspace'].to(device)     # [B, Nc, H, W, 2]
        min_val = batch['img_min'].to(device) # Renamed from 'min' to avoid conflict with built-in
        max_val = batch['img_max'].to(device) # Renamed from 'max' to avoid conflict with built-in

        
        if i < min_images:
            continue
        if i >= max_images:
            break
        
        logger.info(f"Inference for image {i}")
        fname = f'{i:03}.png'
        
        try:
            # Ensure ref_img is on the correct device
            ref_img = ref_img.to(device)
            
            # Log original reference image to TensorBoard
            if writer is not None:
                # Normalize image for display
                normalized_ref = (ref_img[0] + 1) / 2  # Convert from [-1,1] to [0,1]
                writer.add_image(f'Original/Image_{i}', normalized_ref, i)
            
            
            # Special case handling: MRI
            if measure_config['operator']['name'] == 'mri_acce':
                try:
                    # Construct MRI specific conditioning function, bind necessary parameters
                    measurement_cond_fn = partial(
                        cond_method.conditioning,
                        mask=mask,
                        csm=csm,
                        img_min=min_val,
                        img_max=max_val,
                    )

                    # Bind to sample_fn, so it uses these parameters on default call
                    sample_fn_specific = partial(sample_fn, measurement_cond_fn=measurement_cond_fn) # Use a new var name to avoid rebinding in loop

                    # MRI does not require additional forward simulation, as k_under is already provided
                    # y_n is the directly passed under_image_rec, if noise simulation is needed, add it here
                    y_n = operator.forward(ref_img, mask, csm, min_val, max_val)
                    
                except Exception as e:
                    logger.error(f"Error processing MRI: {e}")
                    continue

            # Special case handling: inpainting
            elif measure_config['operator']['name'] == 'inpainting': # Changed to elif
                try:
                    # Generate mask
                    current_mask = mask_gen(ref_img) # Renamed to current_mask
                    current_mask = current_mask[:, 0, :, :].unsqueeze(dim=0)
                    
                    # Update conditioning function
                    measurement_cond_fn = partial(cond_method.conditioning, mask=current_mask)
                    sample_fn_specific = partial(sample_fn, measurement_cond_fn=measurement_cond_fn) # Use a new var name
                    
                    # Apply measurement model (Ax + n)
                    y = operator.forward(ref_img, mask=current_mask)
                    y_n = noiser(y.to(device))
                    
                    # Log mask to TensorBoard
                    if writer is not None:
                        writer.add_image(f'Masks/Image_{i}', current_mask[0], i)
                except Exception as e:
                    logger.error(f"Error processing inpainting: {e}")
                    continue
                    
            # Special case handling: turbulence
            elif measure_config['operator']['name'] == 'turbulence':
                try:
                    current_mask = None # Mask is not used here explicitly for y_n generation
                    img_size = ref_img.shape[-1]
                    tilt = generate_tilt_map(img_h=img_size, img_w=img_size, kernel_size=7, device=device)
                    tilt = torch.clip(tilt, -2.5, 2.5)
                    kernel_size_val = task_config.get("kernel_size", 31)  # Use default value to prevent KeyError
                    intensity = task_config.get("intensity", 3.0)
                    
                    # Blur kernel
                    conv = Blurkernel('gaussian', kernel_size=kernel_size_val, device=device, std=intensity)
                    kernel = conv.get_kernel().type(torch.float32)
                    kernel = kernel.to(device).view(1, 1, kernel_size_val, kernel_size_val)
                    y = operator.forward(ref_img, kernel, tilt)
                    
                    y_n = noiser(y).to(device)
                    
                    # Log turbulence map and kernel to TensorBoard
                    if writer is not None:
                        if tilt[0][0].dim() == 2:  # Ensure dimensions are correct
                            writer.add_image(f'Turbulence/Tilt_Map_{i}', tilt[0][0].unsqueeze(0), i)
                        if kernel[0][0].dim() == 2:  # Ensure dimensions are correct
                            writer.add_image(f'Turbulence/Kernel_{i}', kernel[0][0].unsqueeze(0), i)
                except Exception as e:
                    logger.error(f"Error processing turbulence: {e}")
                    continue
            else: # Default case if not mri_acce, inpainting or turbulence
                # This assumes a generic forward pass if specific conditions aren't met.
                # You might need to adjust this based on how 'mask' and 'csm' (which is now current_mask)
                # are used in your generic operator.forward.
                # If 'y_n' is not supposed to be calculated here, this block can be removed.
                # y = operator.forward(ref_img, mask=current_mask if 'current_mask' in locals() else None) # Or pass original 'mask' if appropriate
                # y_n = noiser(y.to(device))
                # For now, let's assume y_n should be pre-defined or handled by the algo if not a special case
                pass # Or handle generic case explicitly to define y_n

            # Set a fixed random seed for each image
            random_seed = 42 + i  # Use a different seed for each image
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            # torch.backends.cudnn.deterministic = True # Can slow down, enable if strict reproducibility is critical
            # torch.backends.cudnn.benchmark = False

            # Log measurement results to TensorBoard
            if writer is not None and 'y_n' in locals() and y_n is not None: # Check if y_n is defined
                normalized_y_n = (y_n[0] + 1) / 2  # Convert from [-1,1] to [0,1]
                writer.add_image(f'Measurements/Image_{i}', normalized_y_n, i)
            
            # Execute processing according to the selected algorithm
            start_time = time.time()
            
            # Determine the correct sample_fn and measurement_cond_fn to use
            current_sample_fn = sample_fn_specific if 'sample_fn_specific' in locals() and measure_config['operator']['name'] in ['mri_acce', 'inpainting'] else sample_fn
            current_measurement_cond_fn = measurement_cond_fn # This might need to be more specific if not set by MRI/inpainting

            if args.algo == 'dmplug':
                try:
                    sample, metrics = DMPlug(
                        model, sampler, current_measurement_cond_fn, ref_img, y_n, args, operator, device, model_config,
                        measure_config, fname, early_stopping_threshold=1e-3, stop_patience=15, out_path=out_path,
                        iteration=args.iter, lr=0.05, denoiser_step=args.timestep, mask=(current_mask if measure_config['operator']['name'] == 'inpainting' else mask), random_seed=random_seed,
                        writer=writer, img_index=i
                    )
                except Exception as e:
                    logger.error(f"DMPlug execution error: {e}")
                    continue
                
            elif args.algo == 'dmplug_turbulence':
                try:
                    sample, metrics = DMPlug_turbulence(
                        model, sampler, current_measurement_cond_fn, ref_img, y_n, args, operator, device, model_config,
                        measure_config, task_config, fname, kernel_ref=kernel, early_stopping_threshold=1e-3, 
                        stop_patience=5, out_path=out_path, iteration=args.iter, lr=0.05, denoiser_step=args.timestep, 
                        mask=None, random_seed=random_seed, writer=writer, img_index=i # Turbulence might not use 'mask' in the same way
                    )
                except Exception as e:
                    logger.error(f"DMPlug_turbulence execution error: {e}")
                    continue
                
            elif args.algo == 'mpgd':
                try:
                    # MRI special handling: duplicate single-channel image to 3 channels
                    temp_ref_img = ref_img
                    temp_y_n = y_n
                    if measure_config['operator']['name'] == 'mri_acce':
                        if ref_img.shape[1] == 1:
                            temp_ref_img = ref_img.repeat(1, 3, 1, 1)
                        if y_n.shape[1] == 1:
                            temp_y_n = y_n.repeat(1, 3, 1, 1)

                        sample, metrics = mpgd_mri( # Assuming mpgd_mri is specifically for 3-channel
                            current_sample_fn, temp_ref_img, temp_y_n, out_path, fname, device, 
                            mask=mask, random_seed=random_seed, writer=writer, img_index=i # Original mask from dataloader for MRI
                        )
                    else:
                        sample, metrics = mpgd(
                                current_sample_fn, temp_ref_img, temp_y_n, out_path, fname, device, 
                                mask=(current_mask if measure_config['operator']['name'] == 'inpainting' else None), random_seed=random_seed, writer=writer, img_index=i
                                )
                except Exception as e:
                    logger.error(f"mpgd execution error: {e}")
                    continue
                

            elif args.algo == 'acce_RED_diff':
                try:
                    sample, metrics, psnr_curve = acce_RED_diff(
                        model, sampler, current_measurement_cond_fn, ref_img, y_n, device, model_config, measure_config, operator, fname,
                        iter_step=int(args.iter_step), iteration=args.iter, denoiser_step=args.timestep, stop_patience=15, 
                        early_stopping_threshold=0.02, lr=args.lr, out_path=out_path, mask=(current_mask if measure_config['operator']['name'] == 'inpainting' else None), random_seed=random_seed,
                        writer=writer, img_index=i
                    )
                    # Add current image's psnr_curve to the list of all curves
                    all_psnr_curves.append({
                        'image_index': i,
                        'filename': fname,
                        'psnr_curve': psnr_curve['psnrs']
                    })
                    
                except Exception as e:
                    logger.error(f"acce_RED_diff execution error: {e}")
                    continue
                # Convert numpy array to list for JSON serialization
                for curve_data in all_psnr_curves:
                    if isinstance(curve_data['psnr_curve'], np.ndarray):
                        curve_data['psnr_curve'] = curve_data['psnr_curve'].tolist()

                # Save to JSON file
                with open(os.path.join(out_path, 'all_psnr_curves.json'), 'w') as f:
                    json.dump(all_psnr_curves, f, indent=4)
            
            elif args.algo == 'dps_mri':
                try:
                    sample, metrics = dps_mri(
                        current_sample_fn, ref_img, y_n, out_path, fname, device, mask, random_seed=random_seed, # Original mask for MRI
                        writer=writer, img_index=i
                    )
                except Exception as e:
                    logger.error(f"dps_mri execution error: {e}") # Changed from dps to dps_mri
                    logger.error(traceback.format_exc())  # Print full stack trace
                    continue
                
            elif args.algo == 'acce_RED_diff_mri':
                try:
                    sample, metrics, psnr_curve = acce_RED_diff_mri(
                        model, sampler, current_measurement_cond_fn, ref_img, y_n, k_under, mask,csm,min_val, max_val , device, model_config, measure_config, operator, fname,
                        iter_step=int(args.iter_step), iteration=args.iter, denoiser_step=args.timestep, stop_patience=15, 
                        early_stopping_threshold=0.02, lr=args.lr, out_path=out_path, random_seed=random_seed,
                        writer=writer, img_index=i
                    )
                    # Add current image's psnr_curve to the list of all curves
                    all_psnr_curves.append({
                        'image_index': i,
                        'filename': fname,
                        'psnr_curve': psnr_curve['psnrs']
                    })
                    
                except Exception as e:
                    logger.error(f"acce_RED_diff_mri execution error: {e}") # Changed from acce_RED_diff
                    traceback.print_exc()  # <<< Add this!
                    continue
                # Convert numpy array to list for JSON serialization
                for curve_data in all_psnr_curves:
                    if isinstance(curve_data['psnr_curve'], np.ndarray):
                        curve_data['psnr_curve'] = curve_data['psnr_curve'].tolist()

                # Save to JSON file
                with open(os.path.join(out_path, 'all_psnr_curves.json'), 'w') as f:
                    json.dump(all_psnr_curves, f, indent=4)
                    
        
            elif args.algo == 'acce_RED_diff_ablation':
                try:
                    sample, metrics, psnr_curve = acce_RED_diff_ablation(
                        model, sampler, current_measurement_cond_fn, ref_img, y_n, device, model_config, measure_config, operator, fname,
                        iter_step=int(args.iter_step), iteration=args.iter, denoiser_step=args.timestep, stop_patience=15, 
                        early_stopping_threshold=0.02, lr=args.lr, out_path=out_path, mask=(current_mask if measure_config['operator']['name'] == 'inpainting' else None), random_seed=random_seed,
                        writer=writer, img_index=i
                    )
                    # Add current image's psnr_curve to the list of all curves
                    all_psnr_curves.append({
                        'image_index': i,
                        'filename': fname,
                        'psnr_curve': psnr_curve['psnrs']
                    })
                    
                except Exception as e:
                    logger.error(f"acce_RED_diff_ablation execution error: {e}")
                    continue
                # Convert numpy array to list for JSON serialization
                for curve_data in all_psnr_curves:
                    if isinstance(curve_data['psnr_curve'], np.ndarray):
                        curve_data['psnr_curve'] = curve_data['psnr_curve'].tolist()

                # Save to JSON file
                with open(os.path.join(out_path, 'all_psnr_curves.json'), 'w') as f:
                    json.dump(all_psnr_curves, f, indent=4)
                            
            elif args.algo == 'acce_RED_diff_turbulence':
                try:
                    sample, metrics = acce_RED_diff_turbulence(
                        model, sampler, current_measurement_cond_fn, ref_img, y_n, device, model_config, measure_config, task_config, operator, fname,
                        kernel_ref=kernel, iter_step=3, iteration=args.iter, denoiser_step=args.timestep, stop_patience=5, 
                        early_stopping_threshold=0.02, lr=0.01, out_path=out_path, mask=None, random_seed=random_seed, # Turbulence specific mask handling
                        writer=writer, img_index=i
                    )
                except Exception as e:
                    logger.error(f"acce_RED_diff_turbulence execution error: {e}")
                    continue
            
            elif args.algo == 'dps':
                try:
                    sample, metrics = DPS(
                        current_sample_fn, ref_img, y_n, out_path, fname, device, mask=(current_mask if measure_config['operator']['name'] == 'inpainting' else mask), random_seed=random_seed, # Use specific mask if inpainting
                        writer=writer, img_index=i
                    )
                except Exception as e:
                    logger.error(f"dps execution error: {e}")
                    continue
 
            else:
                logger.error(f"Unknown algorithm: {args.algo}")
                continue
                
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            logger.info(f"{args.algo} execution time: {execution_time:.2f} seconds")
            
            # Log reconstruction results to TensorBoard
            if writer is not None:
                # Ensure the sample shape is correct
                if sample.dim() == 4:
                    sample_img = sample[0]
                else:
                    sample_img = sample
                
                normalized_sample = (sample_img + 1) / 2  # Convert from [-1,1] to [0,1]
                writer.add_image(f'Reconstructions/Image_{i}', normalized_sample, i)
                
                # Log error map (difference between original and reconstructed image)
                error_map = torch.abs(ref_img - sample) # Ensure ref_img is comparable to sample (e.g. if ref_img was temp_ref_img)
                if error_map.dim() == 4:
                    error_map = error_map[0]
                # Normalize error map for display
                error_map_max = error_map.max()
                if error_map_max > 1e-8: # Avoid division by zero or very small numbers
                    error_map = error_map / error_map_max
                else:
                    error_map = torch.zeros_like(error_map) # Or handle as appropriate
                writer.add_image(f'ErrorMaps/Image_{i}', error_map, i)
            
            # Log returned metrics
            if metrics:
                psnrs_list.append(metrics.get('psnr', 0))
                ssims_list.append(metrics.get('ssim', 0))
                lpipss_list.append(metrics.get('lpips', 0))
                
                # Log to CSV
                with open(out_csv_path, mode='a', newline='') as csv_file:
                    csvwriter_loop = csv.writer(csv_file) # Renamed to avoid conflict
                    csvwriter_loop.writerow([fname, metrics.get('psnr', 0), metrics.get('ssim', 0), metrics.get('lpips', 0)])
                
                # Log metrics for each image in TensorBoard
                if writer is not None:
                    writer.add_scalar('Metrics/PSNR_per_image', metrics.get('psnr', 0), i)
                    writer.add_scalar('Metrics/SSIM_per_image', metrics.get('ssim', 0), i)
                    writer.add_scalar('Metrics/LPIPS_per_image', metrics.get('lpips', 0), i)
            
        except Exception as e:
            logger.error(f"Error processing image {i}: {e}")
            logger.error(traceback.format_exc())
            continue
            
    # After processing all images, log average metrics
    if psnrs_list:
        avg_psnr = np.mean(psnrs_list)
        avg_ssim = np.mean(ssims_list)
        avg_lpips = np.mean(lpipss_list)
        std_psnr = np.std(psnrs_list)
        std_ssim = np.std(ssims_list)
        std_lpips = np.std(lpipss_list)
        
        if writer is not None:
            writer.add_scalar('Metrics/Avg_PSNR', avg_psnr, 0)
            writer.add_scalar('Metrics/Avg_SSIM', avg_ssim, 0)
            writer.add_scalar('Metrics/Avg_LPIPS', avg_lpips, 0)
            writer.add_scalar('Metrics/Std_PSNR', std_psnr, 0)
            writer.add_scalar('Metrics/Std_SSIM', std_ssim, 0)
            writer.add_scalar('Metrics/Std_LPIPS', std_lpips, 0)
        
        # Display metric distribution using box plots
        try:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].boxplot(psnrs_list)
            ax[0].set_title('PSNR Distribution')
            ax[1].boxplot(ssims_list)
            ax[1].set_title('SSIM Distribution')
            ax[2].boxplot(lpipss_list)
            ax[2].set_title('LPIPS Distribution')
            plt.tight_layout()
            
            # Save to local file and add to TensorBoard
            dist_path = os.path.join(out_path, 'metric_distributions.png')
            plt.savefig(dist_path)
            
            if writer is not None and os.path.exists(dist_path):
                # Read image using a method that doesn't require GUI backend
                img_array = plt.imread(dist_path)
                if img_array.ndim == 3 and img_array.shape[2] == 4: # RGBA
                    img_array = img_array[:, :, :3] # Convert to RGB
                img = torchvision.transforms.ToTensor()(img_array)
                writer.add_image('Distributions/Boxplots', img, 0)
            plt.close(fig) # Close the specific figure
            
            # Log average execution time
            avg_execution_time_val = np.mean(execution_times) if execution_times else 0 # Renamed
            if writer is not None:
                writer.add_scalar('Performance/Avg_Execution_Time', avg_execution_time_val, 0)
            
            # Plot PSNR, SSIM, LPIPS values for images as bar charts
            fig_metrics, ax_metrics = plt.subplots(1, 3, figsize=(15, 5)) # Renamed fig and ax
            x_vals = list(range(len(psnrs_list))) # Renamed
            ax_metrics[0].bar(x_vals, psnrs_list)
            ax_metrics[0].set_title('PSNR per Image')
            ax_metrics[0].set_xlabel('Image Index')
            ax_metrics[0].set_ylabel('PSNR (dB)')
            
            ax_metrics[1].bar(x_vals, ssims_list)
            ax_metrics[1].set_title('SSIM per Image')
            ax_metrics[1].set_xlabel('Image Index')
            ax_metrics[1].set_ylabel('SSIM')
            
            ax_metrics[2].bar(x_vals, lpipss_list)
            ax_metrics[2].set_title('LPIPS per Image')
            ax_metrics[2].set_xlabel('Image Index')
            ax_metrics[2].set_ylabel('LPIPS')
            
            plt.tight_layout()
            metrics_path = os.path.join(out_path, 'metrics_per_image.png')
            plt.savefig(metrics_path)
            
            if writer is not None and os.path.exists(metrics_path):
                img_array_metrics = plt.imread(metrics_path)
                if img_array_metrics.ndim == 3 and img_array_metrics.shape[2] == 4: # RGBA
                    img_array_metrics = img_array_metrics[:, :, :3] # Convert to RGB
                img_metrics = torchvision.transforms.ToTensor()(img_array_metrics)
                writer.add_image('Distributions/Metrics_per_Image', img_metrics, 0)
            plt.close(fig_metrics) # Close the specific figure
            
        except Exception as e:
            logger.error(f"Error generating metric visualizations: {e}")
            logger.error(traceback.format_exc()) # Added for more details
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        logger.info("TensorBoard writer closed")
    
    # Calculate and output total execution time
    total_execution_time = sum(execution_times)
    avg_execution_time_final = np.mean(execution_times) if execution_times else 0 # Renamed
    logger.info(f"Total execution time ({len(execution_times)} images): {total_execution_time:.2f} seconds")
    logger.info(f"Average execution time per image: {avg_execution_time_final:.2f} seconds")

    # Print final average metrics
    if psnrs_list: # Check if lists are not empty
        logger.info(f"Average PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}")
        logger.info(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
        logger.info(f"Average LPIPS: {avg_lpips:.4f} ± {std_lpips:.4f}")
    
    # Output TensorBoard viewing command
    logger.info(f"\nTo view TensorBoard logs, run:")
    logger.info(f"tensorboard --logdir=\"{os.path.join(out_path, 'tensorboard_logs')}\"") # Added quotes for path

if __name__ == '__main__':
    main()