import argparse
import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import re
from tqdm import tqdm
import numpy as np
from torchvision import models, transforms as T
from scipy import linalg
from transformers import CLIPTokenizer
from chexpert_dataset import CheXpertDataset
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers import DDPMScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images and compute FID using multiple LoRA-finetuned Stable Diffusion models.")

    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_weights_paths", type=str, nargs='+', required=True, help="One or more LoRA weights directories")
    parser.add_argument("--val_data_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, nargs="+")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=7.5)

    return parser.parse_args()


def load_pipeline(args, dtype):
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant
    )
    pipe.to(args.device)
    pipe.enable_attention_slicing()
    return pipe


def generate_images(pipe, args, generator, lora_path, lora_step, output_dir):
    pipe.load_lora_weights(lora_path)
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for i, prompt in enumerate(args.prompt):
            for j in range(args.num_images):
                print(f"Generating image {j+1} for prompt: {prompt}")

                prompt_embeds, neg_embeds = pipe.encode_prompt(prompt, device=pipe._execution_device, num_images_per_prompt=1, do_classifier_free_guidance=True)
                prompt_embeds = torch.cat([neg_embeds, prompt_embeds])

                timesteps, _ = retrieve_timesteps(pipe.scheduler, args.num_inference_steps, pipe._execution_device)

                latents = torch.randn((1, pipe.unet.config.in_channels, 64, 64), generator=generator, device=pipe._execution_device, dtype=pipe.unet.dtype)
                latents *= pipe.scheduler.init_noise_sigma

                memorization_scores = []
                for t in timesteps:
                    latent_input = torch.cat([latents] * 2)
                    latent_input = pipe.scheduler.scale_model_input(latent_input, t)

                    noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
                    uncond, cond = noise_pred.chunk(2)
                    noise_pred = uncond + args.guidance_scale * (cond - uncond)

                    dt = torch.norm(cond - uncond, p=2)
                    memorization_scores.append(dt.item())

                    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
                image = (image / 2 + 0.5).clamp(0, 1)
                image = pipe.image_processor.postprocess(image, output_type="pil")[0]

                prompt_clean = re.sub(r'\W+', '_', prompt.lower())
                image_path = os.path.join(output_dir, f"{lora_step}_{prompt_clean}_img{j+1}.png")
                image.save(image_path)

                d = sum(memorization_scores) / len(memorization_scores)
                print("Memorization detection score:", d)
                with open(os.path.join(output_dir, f"{lora_step}_memorization_scores.txt"), "a") as f:
                    f.write(f"{prompt_clean}_img{j+1}: {d:.4f}\n")

                # Save full time-series of scores for this image
                scores_path = os.path.join(output_dir, f"{lora_step}_{prompt_clean}_img{j+1}_memo_scores.npy")
                np.save(scores_path, np.array(memorization_scores))
                print()


def compute_fid(args, lora_step, output_dir):
    class VGG(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = models.vgg11(weights=models.VGG11_Weights.DEFAULT).features[:10]
            self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
            self.flatten = torch.nn.Flatten()

        def forward(self, x, features=False):
            feat = self.features(x)
            feat = self.avg_pool(feat)
            return feat if features else self.flatten(feat)

    def feature_statistics(features):
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def frechet_distance(mu1, sigma1, mu2, sigma2):
        covmean = linalg.sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean): covmean = covmean.real
        return np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2 * covmean)

    tokenizer = CLIPTokenizer.from_pretrained(args.base_model_path, subfolder="tokenizer")
    val_dataset = CheXpertDataset(data_dir=args.val_data_dir, csv_name="valid.csv", tokenizer=tokenizer)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = VGG().to(args.device).eval()
    feats = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Extracting real features"):
            images = batch["pixel_values"].to(args.device)
            images = torch.nn.functional.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
            images = (images - 0.5) / 0.5
            f = model(images, features=True).squeeze(-1).squeeze(-1)
            feats.append(f.cpu().numpy())
    real_features = np.concatenate(feats, axis=0)

    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    feats = []
    with torch.no_grad():
        for fname in tqdm(sorted(os.listdir(output_dir)), desc="Extracting generated features"):
            if not fname.endswith(".png") or not fname.startswith(f"{lora_step}_"): continue
            image = Image.open(os.path.join(output_dir, fname)).convert("RGB")
            image = transform(image).unsqueeze(0).to(args.device)
            f = model(image, features=True).squeeze().cpu().numpy()
            feats.append(f)
    gen_features = np.stack(feats)

    mu_real, sigma_real = feature_statistics(real_features)
    mu_gen, sigma_gen = feature_statistics(gen_features)
    fid = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    print(f"\nFID score: {fid:.2f}")

    with open(os.path.join(output_dir, "evaluation_results.txt"), "a") as f:
        f.write(f"LoRA checkpoint: {lora_step}, FID score: {fid:.4f}\n")


def main():
    args = parse_args()

    # Override prompt list programmatically
    label_cols = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "Pleural Effusion", "Pleural Other",
        "Pneumonia", "Pneumothorax", "Support Devices", "No Finding"
    ]

    view_types = ["frontal", "lateral"]
    device_options = ["with support devices", "without support devices"]

    # Generate prompts
    prompts = []
    for view in view_types:
        for label in label_cols:
            for device in device_options:
                disease_text = label if label != "No Finding" else "no findings"
                prompt = f"A {view} view of a chest X-ray with {disease_text} {device}"
                prompts.append(prompt)

    args.prompt = prompts

    dtype_map = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.mixed_precision]

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    pipe = load_pipeline(args, dtype)

    for lora_path in args.lora_weights_paths:
        lora_folder = os.path.basename(lora_path)
        match = re.search(r"(\d+)", lora_folder)
        lora_step = match.group(1) if match else "unknown"
        output_dir = os.path.join(os.path.dirname(lora_path), "generated_images")

        generate_images(pipe, args, generator, lora_path, lora_step, output_dir)
        compute_fid(args, lora_step, output_dir)


if __name__ == "__main__":
    main()
