#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import math
import os
import random
import shutil
from contextlib import nullcontext

import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
# Lora things
from peft import LoraConfig
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler

from chexpert_dataset import CheXpertDataset





def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, step):
    print(f"Running validation... \n Generating {args.num_validation_images} images with prompt: {args.validation_prompt}.")
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    for _ in range(args.num_validation_images):
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)

        with autocast_ctx:
            image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
        images.append(image)

    del pipeline
    torch.cuda.empty_cache()
    # Save images
    output_dir = os.path.join(args.output_dir, f"validation_images/step_{step}")
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(images):
        img.save(os.path.join(output_dir, f"{i}.png"))

    return images


def save_progress(unet, accelerator, args, save_path, safe_serialization=True):
    print("Saving LoRA weights only")
    unet = accelerator.unwrap_model(unet)
    lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=save_path,
        unet_lora_layers=lora_state_dict,
        safe_serialization=safe_serialization,
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Training script hyperparameters.")

    parser.add_argument("--save_steps", type=int, default=500, help="Save learned_embeds.bin every X update steps.")
    parser.add_argument("--save_as_full_pipeline", action="store_true", help="Save the complete stable diffusion pipeline.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Path or model ID from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, help="Optional revision of the pretrained model.")
    parser.add_argument("--variant", type=str, default=None, help="Variant of model files (e.g. 'fp16').")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Tokenizer name/path if different from model.")
    parser.add_argument("--unet_name", type=str, default=None, help="NOT IMPLEMENTED YET! Unet name/path if different from model.") # TODO

    parser.add_argument("--train_data_dir", type=str, required=True, help="Folder containing the training data.")

    parser.add_argument("--output_dir", type=str, default=None, help="Where to save outputs and checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible training.")

    parser.add_argument("--resolution", type=int, default=512, help="Input image resolution (images will be resized).") #TODO implement this in cheXpert (hardlocked 512)
    parser.add_argument("--center_crop", action="store_true", help="Whether to center crop before resizing.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size per device for training.")
    parser.add_argument("--max_train_steps", type=int, default=5000, help="will decide the number of training epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients every X steps.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory.")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--scale_lr", action="store_true", help="Scale LR by GPUs, batch size, and grad accumulation.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="LR scheduler type (e.g. linear, cosine, constant).")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Warmup steps for the LR scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of cycles for cosine_with_restarts scheduler.")

    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Use mixed precision training.")
    parser.add_argument("--rank", type=int, default=4, help=("The dimension of the LoRA update matrices."))

    parser.add_argument("--validation_prompt", type=str, default=None, help="Prompt used during validation.")
    parser.add_argument("--num_validation_images", type=int, default=4, help="Num images to generate during validation.")
    parser.add_argument("--validation_steps", type=int, default=100, help="Validate every X steps.")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank.")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save training state every X steps.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to keep.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path or 'latest' to resume training from.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")
    if args.output_dir is None:
        raise ValueError("You must specify an output folder for saving checkpoints and generated images.")


    return args




def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )


    # This set logging levels to be appropriate when training on 1 GPU
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    # set train seed
    set_seed(args.seed)

    # make the output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Save hyperparameters to a file in the output_dir
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        import json
        json.dump(vars(args), f, indent=4)
    
    # Create an instance of wandb
    run_name = f"{args.output_dir}"

    wandb.init(
        project="DL CV project",            # Your wandb project name (can be anything)
        name=run_name,
        config=vars(args),                # Logs all your hyperparams
        dir=args.output_dir               # Save wandb logs to this directory
    )

    
    # Load tokenizer. If already created use the output path
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant)

    
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    # Freeze vae, text encoder and UNET (but not LORA).
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Do the LORA things
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    # 💡 Inject LoRA into the cross-attention layers of the UNet - Now we can train lora instead of UNET
    unet.add_adapter(unet_lora_config) # Adds LORA to queries, keys, values and out of each block of cross-attention
    lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
    
    print(f"Total trainable LoRA parameters: {sum(p.numel() for p in lora_layers)}")

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        lora_layers,  # only optimize the unet
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # Dataset and DataLoaders creation:
    train_dataset = CheXpertDataset(
        data_dir=args.train_data_dir,
        csv_name="train.csv",
        tokenizer=tokenizer
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        num_workers=0
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    unet.train()
    # Prepare everything with our `accelerator` 
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, text_encoder) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text encoder to device and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # Determine checkpoint path
        def extract_step(name):
            try:
                return int(name.split("-")[-1])
            except:
                return -1

        if args.resume_from_checkpoint != "latest":
            path = args.resume_from_checkpoint
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=extract_step)
            path = dirs[-1] if dirs else None

        if path is None or not os.path.isdir(os.path.join(args.output_dir, path)):
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            initial_global_step = 0
        else:
            checkpoint_path = os.path.join(args.output_dir, path)
            accelerator.print(f"Resuming full training state from {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            global_step = extract_step(path)
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0



    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                images = []
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    weight_name = (f"learned_embeds-steps-{global_step}.safetensors")
                    save_path = os.path.join(args.output_dir, f"lora-steps-{global_step}")

                    save_progress(
                        unet,
                        accelerator,
                        args,
                        save_path,
                        safe_serialization=True
                    )

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                print(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        log_validation(
                            text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            wandb.log(logs, step=global_step)
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    save_full_model = args.save_as_full_pipeline
    if save_full_model:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
        )
        pipeline.save_pretrained(args.output_dir)
    # Save the newly trained embeddings
    weight_name = "unet_final.safetensors"
    save_path = os.path.join(args.output_dir, f"lora-steps-{global_step}")
    save_progress(
        unet,
        accelerator,
        args,
        save_path,
        safe_serialization=True
    )

    wandb.finish()
    accelerator.end_training()


if __name__ == "__main__":
    main()
