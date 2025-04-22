#!/bin/sh 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J post_processing
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
### -- Specify we only want 1 host machine
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=3GB]"
### -- specify which model to use -- This is unnecessary in this server as we only have 1 model
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 4GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 3:40
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo post_processing.out 
#BSUB -eo post_processing.err 
### Necessary bsub option for GPU
#BSUB -gpu "num=1:mode=exclusive_process"

module load python3/3.11.10
source ~/myenv311/bin/activate

python generate_images_unet.py \
    --base_model_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --unet_weights_paths "simplePrompt_full_ageSex/unet-steps-10000" \
    --val_data_dir "/dtu/blackhole/1d/214141/CheXpert-v1.0-small" \
    --num_images 1 \
    --num_inference_steps 50 \
    --seed 42 \
    --mixed_precision fp16 \
    --device cuda \
    --guidance_scale 7.5

python mem_scan_sscd.py \
  --train_data_dir "/dtu/blackhole/1d/214141/CheXpert-v1.0-small" \
  --gen_im_dir "simplePrompt_full_ageSex/generated_images" \
  --batch_size 4 \
  --image_size 512 \
  --output_path "simplePrompt_full_ageSex/best_train_matches.npz"