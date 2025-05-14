#!/bin/bash


# echo "Running experiment with algo=acce_RED_diff"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/super_resolution_config.yaml \
#     --timestep=6 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=400\
#     --noise_type="shot"\
#     --noise_scale=60
# echo "Finished experiment with algo=acce_RED_diff, task = super_resolution"
# echo "----------------------------------------"


# echo "Running experiment with algo=acce_RED_diff"   ### lr :0.01,  p = 1.5 
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \ 
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \ 
#     --task_config=configs/super_resolution_config.yaml \ 
#     --timestep=6 \ 
#     --scale=1 \ 
#     --method="mpgd_wo_proj" \ 
#     --algo="acce_RED_diff" \ 
#     --iter=500 \
#     --noise_type="gaussian" \ 
#     --noise_scale=0

# echo "Finished experiment with algo=acce_RED_diff, task = super_resolution"
# echo "----------------------------------------"

# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/gaussian_deblur_config.yaml \
#     --timestep=6 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=500\
#     --noise_type="gaussian"\
#     --noise_scale=0.01
# echo "Finished experiment with algo=acce_RED_diff, task = gaussian_deblur"
# echo "----------------------------------------"

# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/motion_deblur_config.yaml \
#     --timestep=6 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=500\
#     --noise_type="gaussian"\
#     --noise_scale=0.0
# echo "Finished experiment with algo=acce_RED_diff, task = motion_deblur"
# echo "----------------------------------------"

# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/nonlinear_deblur_config.yaml \
#     --timestep=6 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=20\
#     --noise_type="gaussian"\
#     --noise_scale=0.
# echo "Finished experiment with algo=acce_RED_diff, task = nonlinear_deblur"
# echo "----------------------------------------"

# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/inpainting_config.yaml \
#     --timestep=6 \
#     --scale=1 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff"\
#     --iter=500\
#     --noise_type="gaussian"\
#     --noise_scale=0
# echo "Finished experiment with algo=acce_RED_diff, task = inpainting"
# echo "----------------------------------------"

# echo "Running experiment with algo=acce_RED_diff_turbulence"
# python ffhq_sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/mgpd_diffusion_config.yaml \
#     --task_config=configs/turbulence_config.yaml \
#     --timestep=6 \
#     --scale=17.5 \
#     --method="mpgd_wo_proj" \
#     --algo="acce_RED_diff_turbulence"\
#     --iter=400\
#     --noise_type="gaussian"\
#     --noise_scale=0.01
# echo "Finished experiment with algo=acce_RED_diff_turbulence, task = bid_turbulence"
# echo "----------------------------------------"



############  MRI reconstruction 

echo "Running experiment with algo=acce_RED_diff"   ### lr :0.01,  p = 1.5  
python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/mri_acceleration_config.yaml \
    --timestep=10 \
    --scale=1 \
    --method="mpgd_wo_proj" \
    --algo="acce_RED_diff"\
    --iter=50\
    --noise_type="gaussian"\
    --noise_scale=0.01

echo "Finished experiment with algo=acce_RED_diff, task = mri_acceleration"
echo "----------------------------------------"
