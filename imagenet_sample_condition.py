from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt # Only imsave is used, but this is fine.

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.guided_gaussian_diffusion import create_sampler
# from data.dataloader import get_dataset, get_dataloader # get_dataset is not used
from data.dataloader import get_dataloader # Only get_dataloader is used from this
from util.img_utils import clear_color # mask_generator is imported but not used
from util.logger import get_logger
import torchvision


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser(description="Image reconstruction using guided diffusion.") # Added a description
    parser.add_argument('--model_config', type=str, default="", help="Path to the model configuration YAML file.")
    parser.add_argument('--diffusion_config', type=str, required=True, help="Path to the diffusion configuration YAML file.") # Made required
    parser.add_argument('--task_config', type=str, required=True, help="Path to the task configuration YAML file.") # Made required
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to use.")
    parser.add_argument('--timestep', type=int, default=100, help="Number of timesteps for DDIM.")
    parser.add_argument('--eta', type=float, default=0.5, help="Eta parameter for DDIM.")
    parser.add_argument('--scale', type=float, default=10, help="Conditioning scale.")
    parser.add_argument('--method', type=str, default='mpgd_wo_proj', help="Conditioning method.")
    parser.add_argument('--save_dir', type=str, default='./outputs/imagenet/', help="Directory to save output images.")
    parser.add_argument('--imagenet_root', type=str, default='imagenet_root', help="Root directory of the ImageNet dataset.") # Made imagenet_root an arg

    args = parser.parse_args()
    
    # Logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config) if args.model_config else {} # Handle empty string case
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    
    # Update diffusion config based on timestep argument
    if args.timestep < 1000:
        diffusion_config["timestep_respacing"] = f"ddim{args.timestep}"
        diffusion_config["rescale_timesteps"] = True
    else:
        diffusion_config["timestep_respacing"] = "1000" # Ensure it's a string if that's what create_sampler expects
        diffusion_config["rescale_timesteps"] = False
    
    diffusion_config["eta"] = args.eta
    
    # Update task config
    task_config["conditioning"]["method"] = args.method
    task_config["conditioning"]["params"]["scale"] = args.scale
    # Hardcoded dataset name, assuming this script is specific to ImageNet 256
    task_config["data"]["name"] = "imagenet256" 
    
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
    # Note: Checkpoint path is hardcoded, common for specific experiment scripts.
    cond_method = get_conditioning_method(cond_config['method'], 
                                          operator, 
                                          noiser, 
                                          resume="../nonlinear/SD_style/models/ldm/cin256/model.ckpt", 
                                          **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method: {task_config['conditioning']['method']}") # Corrected typo "Conditioning method :"
    
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
    
    # Prepare output directory
    # dir_path_suffix = f"ddim{args.timestep}_eta{args.eta}_scale{args.scale}" # More descriptive suffix
    dir_path_suffix = f"{diffusion_config['timestep_respacing']}_eta{args.eta}_scale{args.scale}"
    out_path = os.path.join(args.save_dir, 
                            measure_config['operator']['name'], 
                            task_config['conditioning']['method'], 
                            dir_path_suffix)
    
    os.makedirs(out_path, exist_ok=True)
    for img_dir_name in ['input', 'recon', 'progress', 'label']: # Renamed loop var for clarity
        os.makedirs(os.path.join(out_path, img_dir_name), exist_ok=True)

    # Prepare dataloader
    # data_config = task_config['data'] # This variable is not used further
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Standard normalization for [-1, 1] range
    ])
    # Ensure ImageNet root is correctly specified by the user
    dataset = torchvision.datasets.ImageFolder(root=args.imagenet_root, transform=transform) 
    logger.info(f"Loading dataset from: {args.imagenet_root}")
    
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False) # num_workers=0 for simplicity, can be increased
        
    # Perform Inference
    logger.info("Starting inference...")
    max_images_per_class_set = 1000 # Total unique classes to process
    processed_classes = set() # Use a set for efficient 'in' check
    images_saved_count = 0

    for i, (ref_img, class_label) in enumerate(loader):
        class_label = int(class_label)
        
        if class_label in processed_classes:
            continue
        
        if images_saved_count >= max_images_per_class_set:
            logger.info(f"Reached the limit of {max_images_per_class_set} unique classes processed.")
            break
        
        processed_classes.add(class_label)
        
        logger.info(f"Processing image {i} (class {class_label}). Total unique classes processed: {len(processed_classes)}.")
        # Filename includes original image index and class label for easier tracking
        fname = f'img{i:05}_class{class_label:05}.png'
        
        ref_img = ref_img.to(device)

        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img)
        y_n = noiser(y)

        # Sampling
        # Initial noise tensor for the sampling process. .requires_grad_() is typically not needed for x_start in p_sample_loop.
        x_start = torch.randn(ref_img.shape, device=device) 
        
        # sample_fn is sampler.p_sample_loop
        # 'record=True' and 'save_root' suggest p_sample_loop might save intermediate steps.
        sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)

        # Save images
        # clear_color function is used before saving, ensure it returns a NumPy array compatible with plt.imsave
        # or a Tensor that's properly handled.
        try:
            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n.cpu()))
            plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img.cpu()))
            plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample.cpu()))
            images_saved_count += 1
        except Exception as e:
            logger.error(f"Failed to save image {fname}. Error: {e}")
            # Decide if you want to continue or break if saving fails
            continue 
            
    logger.info(f"Inference complete. Total images saved: {images_saved_count}.")

if __name__ == '__main__':
    main()