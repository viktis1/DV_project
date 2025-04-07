import argparse
import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import re
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using Hugging Face's default StableDiffusionPipeline.")

    parser.add_argument("--base_model_path", type=str, required=True, help="Path or model ID for the base SD model.")
    parser.add_argument("--lora_weights_path", type=str, required=True, help="Path to LoRA weights.")
    parser.add_argument("--prompt", type=str, nargs="+", required=True, help="Prompts (space-separated list).")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images per prompt.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=7.5)

    return parser.parse_args()


def main():
    args = parse_args()

    dtype_map = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[args.mixed_precision]
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    print("Loading pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant
    )
    pipe.load_lora_weights(args.lora_weights_path)
    pipe.to(args.device)
    pipe.enable_attention_slicing()

    output_dir = os.path.join(os.path.dirname(args.lora_weights_path), "generated_images_baseline")
    os.makedirs(output_dir, exist_ok=True)

    lora_step = re.search(r"(\d+)", os.path.basename(args.lora_weights_path))
    lora_step = lora_step.group(1) if lora_step else "unknown"

    for i, prompt in enumerate(args.prompt):
        print(f"Generating {args.num_images} image(s) for prompt: \"{prompt}\"")
        for j in tqdm(range(args.num_images), desc=f"Prompt {i+1}"):
            image = pipe(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator
            ).images[0]

            prompt_clean = re.sub(r'\W+', '_', prompt.lower())[:40]
            image_path = os.path.join(output_dir, f"{lora_step}_pipe_{prompt_clean}_img{j+1}.png")
            image.save(image_path)
            print(f"Saved: {image_path}")


if __name__ == "__main__":
    main()
