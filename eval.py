import argparse
import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import re # For finding the lora weight checkpoint
from tqdm import tqdm
import numpy as np
from torchvision import models
from scipy import linalg
from transformers import CLIPTokenizer
from chexpert_dataset import CheXpertDataset

# This uses as many functions from the diffuser library as possible to avoid understanding their implementation as much as possible.



def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using a LoRA-finetuned Stable Diffusion model.")

    parser.add_argument("--base_model_path", type=str, required=True, help="Path or model ID for the base SD model (the pretrained one).")
    parser.add_argument("--lora_weights_path", type=str, required=True, help="Path to directory with LoRA weights.")
    parser.add_argument("--val_data_dir", type=str, required=True, help="Folder containing the validation data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no", help="Precision mode.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on.")
    parser.add_argument("--revision", type=str, default=None, help="Model revision.")
    parser.add_argument("--variant", type=str, default=None, help="Variant of the model (e.g., 'fp16').")

    return parser.parse_args()


class VGG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.vgg11(weights=models.VGG11_Weights.DEFAULT).features[:10]
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.flatten = torch.nn.Flatten()

    def forward(self, x, features=False):
        feat = self.features(x)
        feat = self.avg_pool(feat)
        if features:
            return feat
        else:
            return self.flatten(feat)

def feature_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def frechet_distance(mu1, sigma1, mu2, sigma2):
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2*covmean)

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
    
    # Find tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model_path, subfolder="tokenizer")

    # Prepare directory for generated images
    gen_im_dir = os.path.join(os.path.dirname(args.lora_weights_path), "generated_images") # Directory for generated images

    # Find the validation dataset images
    val_dataset = CheXpertDataset(
        data_dir=args.val_data_dir,
        csv_name="valid.csv",
        tokenizer=tokenizer
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
        # ------------------ FID ------------------ #
    print("Preparing VGG for feature extraction...")
    model = VGG().to(args.device)
    model.eval()
    # Define the transform to make images correct dim
    

    # extract_features_from_dataloader
    feats = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Extracting real features"):
            images = batch["pixel_values"].to(args.device)
            images = torch.nn.functional.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
            images = (images - 0.5) / 0.5  # Normalize to match generated image transform
            f = model(images, features=True).squeeze(-1).squeeze(-1)
            feats.append(f.cpu().numpy())
    real_features = np.concatenate(feats, axis=0)

    # extract_features_from_folder 
    feats = []
    from torchvision import transforms as T
    transform = T.Compose([T.Resize((224, 224)),
                           T.ToTensor(),
                           T.Normalize([0.5]*3, [0.5]*3)])
    with torch.no_grad():
        for fname in tqdm(sorted(os.listdir(gen_im_dir)), desc="Extracting generated features"):
            if not fname.endswith(".png"): continue
            if not fname.startswith(f"{lora_step}_"): continue  # 👈 Only keep relevant step
            image = Image.open(os.path.join(gen_im_dir, fname)).convert("RGB")
            image = transform(image).unsqueeze(0).to(args.device)
            f = model(image, features=True).squeeze().cpu().numpy()
            feats.append(f)
    gen_features = np.stack(feats)


    # Compute statistics
    mu_real, sigma_real = feature_statistics(real_features)
    mu_gen, sigma_gen = feature_statistics(gen_features)

    # Compute FID
    fid = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    print(f"\n💥 FID score: {fid:.2f}")

    # Save the FID
    results_dir = os.path.dirname(args.lora_weights_path)
    fid_path = os.path.join(results_dir, "evaluation_results.txt")

    with open(fid_path, "a") as f:
        f.write(f"LoRA checkpoint: {lora_step}, FID score: {fid:.4f}\n")
    




if __name__ == "__main__":
    main()
