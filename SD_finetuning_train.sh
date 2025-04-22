#!/bin/sh 
### -- specify queue -- 
#BSUB -q gpua100
### -- set the job Name -- 
#BSUB -J SD_finetuning_train_sex
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
#BSUB -W 4:00
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo SD_finetuning_train_sex.out 
#BSUB -eo SD_finetuning_train_sex.err 
### Necessary bsub option for GPU
#BSUB -gpu "num=1:mode=exclusive_process"

module load python3/3.11.10
source ~/myenv311/bin/activate

accelerate launch SD_finetuning_train_unet.py \
    --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --train_data_dir "/dtu/blackhole/1d/214141/CheXpert-v1.0-small" \
    --mixed_precision "fp16" \
    --validation_prompt "A chest X-ray image" \
    --output_dir "simplePrompt_full_ageSex" \
    --max_train_steps 10000 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --resume_from_checkpoint latest \
    --checkpoints_total_limit 1 \
    --lr_scheduler cosine \
    --lr_num_cycles 1 
###    --overfit_n 10 \
###    --overfit_repeat 100
###    --threshold_memorization 2
