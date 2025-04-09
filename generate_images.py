import argparse
import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import re # For finding the lora weight checkpoint
from tqdm import tqdm

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

# This uses as many functions from the diffuser library as possible to avoid understanding their implementation as much as possible.



def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using a LoRA-finetuned Stable Diffusion model.")

    parser.add_argument("--base_model_path", type=str, required=True, help="Path or model ID for the base SD model (the pretrained one).")
    parser.add_argument("--lora_weights_path", type=str, required=True, help="Path to directory with LoRA weights.")
    parser.add_argument("--prompt", type=str, nargs="+", required=True, help="One or more prompts to condition image generation (space-separated).")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate.")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no", help="Precision mode.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on.")
    parser.add_argument("--revision", type=str, default=None, help="Model revision.")
    parser.add_argument("--variant", type=str, default=None, help="Variant of the model (e.g., 'fp16').")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for classifier-free guidance.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Convert precision string to torch dtype
    dtype_map = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[args.mixed_precision]

    # Set generator for reproducibility
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant
    )

    # Load LoRA weights
    pipe.load_lora_weights(args.lora_weights_path)


    # Find the lora checkpoint
    lora_folder = os.path.basename(args.lora_weights_path)
    match = re.search(r"(\d+)", lora_folder)
    lora_step = match.group(1) if match else "unknown"

    # Move to device and enable attention slicing
    pipe.to(args.device)
    pipe.enable_attention_slicing()

    # Prepare output directory
    output_dir = os.path.join(os.path.dirname(args.lora_weights_path), "generated_images")
    os.makedirs(output_dir, exist_ok=True)
    # Inside your loop:
    with torch.no_grad():
        for i, prompt in enumerate(args.prompt):
            for j in range(args.num_images):
                print(f"Generating image {j+1} for prompt: {prompt}")

                #  Encode the prompt
                prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                    prompt=prompt,
                    device=pipe._execution_device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=None,
                )
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

                # Set up the timesteps
                timesteps, num_inference_steps = retrieve_timesteps(
                    pipe.scheduler,
                    num_inference_steps=args.num_inference_steps,
                    device=pipe._execution_device
                )

                # Prepare latents (initial noise)
                batch_size = 1
                height, width = 512, 512  # or dynamically set from `pipe.unet.config.sample_size`
                latents = torch.randn(
                    (batch_size, pipe.unet.config.in_channels, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor),
                    generator=generator,
                    device=pipe._execution_device,
                    dtype=pipe.unet.dtype,
                ) * pipe.scheduler.init_noise_sigma
                             
                # Denoising loop with memorization statistic
                T = args.num_inference_steps
                memorization_scores  = []
                for t in timesteps:
                    latent_input = torch.cat([latents] * 2)
                    latent_input = pipe.scheduler.scale_model_input(latent_input, t)

                    noise_pred = pipe.unet(
                        latent_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # save the memorization statistic
                    dt = torch.norm(noise_pred_text - noise_pred_uncond, p=2)
                    memorization_scores.append(dt.item())

                    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Decode latents
                image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
                image = (image / 2 + 0.5).clamp(0, 1)
                image = pipe.image_processor.postprocess(image, output_type="pil")[0]

                # Save
                prompt_clean = re.sub(r'\W+', '_', prompt.lower())[:40]
                image_path = os.path.join(output_dir, f"{lora_step}_{prompt_clean}_img{j+1}.png")
                image.save(image_path)

                # Final memorization metric
                d = sum(memorization_scores) / len(memorization_scores)
                print("Memorization detection score:", d)
                # Save memorization score
                score_path = os.path.join(output_dir, f"{lora_step}_memorization_scores.txt")
                with open(score_path, "a") as f:
                    f.write(f"{prompt_clean}_img{j+1}: {d:.4f}\n")
                print()





if __name__ == "__main__":
    main()
