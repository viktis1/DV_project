from torch.utils.data import DataLoader
import argparse
import torch
import numpy as np
from tqdm import tqdm
from chexpert_dataset import CheXpertDataset
from torchvision import transforms
from PIL import Image
import os
import re
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from itertools import islice
from diffusers import StableDiffusionPipeline
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


# Dummy tokenizer placeholder (GLOBAL)
class DummyOutput:
    def __init__(self):
        self.input_ids = torch.zeros((1, 77), dtype=torch.long)

class DummyTokenizer:
    def __call__(self, *args, **kwargs):
        return DummyOutput()

    @property
    def model_max_length(self):
        return 77

tokenizer = DummyTokenizer()

def parse_args():
    parser = argparse.ArgumentParser(description="Scan for memorization by comparing generated images to training set.")

    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--train_data_dir", type=str, required=True, help="Folder containing the training data.")
    parser.add_argument("--gen_im_dir", type=str, required=True, help="Folder containing generated images.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for loading training data.")
    parser.add_argument("--image_size", type=int, default=512, help="Image size (assumed square) for resizing and comparison.")
    parser.add_argument("--output_path", type=str, default="best_train_matches.npy", help="Where to save the best match indices.")

    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)

    return parser.parse_args()


def load_generated_images(gen_dir, device):
    transform = transforms.ToTensor()
    images = []
    for fname in sorted(os.listdir(gen_dir)):
        if not fname.endswith(".png"):
            continue
        image = Image.open(os.path.join(gen_dir, fname)).convert("RGB")
        image = transform(image).to(device)
        images.append(image)
    return images


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device="cpu" # Slightly faster for me without GPU
    dtype_map = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.mixed_precision]

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant
    )
    pipe.to(device)
    pipe.enable_attention_slicing()
    vae = pipe.vae
    vae.eval()

    # Load generated images
    generated_images = load_generated_images(args.gen_im_dir, device)

    # Load and encode all generated images
    generated_images = load_generated_images(args.gen_im_dir, device)
    gen_tensor = torch.stack(generated_images)
    gen_loader = DataLoader(gen_tensor, batch_size=args.batch_size, shuffle=False)
    gen_latents = []
    for batch in tqdm(gen_loader, desc="Encoding generated images"):
        batch = batch.to(device, dtype=vae.dtype)
        latents = vae.encode(batch).latent_dist.mean.detach()
        gen_latents.append(latents.view(latents.size(0), -1))
    gen_latents = torch.cat(gen_latents, dim=0)  # [N, D]
    gen_latents = torch.nn.functional.normalize(gen_latents, dim=1)
    N = gen_latents.size(0)
    print(f"Stored {N} generated latents.")

    # Load training dataset
    train_dataset = CheXpertDataset(
        data_dir=args.train_data_dir,
        csv_name="train.csv",
        tokenizer=tokenizer,
        image_size=args.image_size
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize best scores, indices and current index (necessary cause working in batches)
    best_scores = torch.full((N,), -float("inf"), dtype=dtype, device=device)
    best_indices = torch.full((N,), -1, dtype=torch.long, device=device)
    current_index = 0
    for batch in tqdm(train_loader, desc="Cosine similarity between gen and train latents"):
        train_imgs = batch["pixel_values"].to(device, dtype=vae.dtype)
        train_latents = vae.encode(train_imgs).latent_dist.mean.detach()
        train_latents = train_latents.view(train_latents.size(0), -1)
        train_latents = torch.nn.functional.normalize(train_latents, dim=1)

        # Compute [N, B] similarity
        sim = torch.matmul(gen_latents, train_latents.T)

        # For each generated image, update best match if current sim is higher
        max_vals, max_idxs = sim.max(dim=1)
        update_mask = max_vals > best_scores
        best_scores[update_mask] = max_vals[update_mask]
        best_indices[update_mask] = current_index + max_idxs[update_mask]

        current_index += train_imgs.size(0)

    # Save results
    np.save(args.output_path, best_indices.cpu().numpy())
    print(f"Saved best match indices to {args.output_path}")


def plot_matches(npy_path, gen_im_dir, train_data_dir, csv_name="train.csv", image_size=512, num_to_plot=50):
    """
    Save side-by-side plots of generated images and their best matching training images.
    """

    import os

    # Output folder for memorization matches
    save_dir = os.path.join(gen_im_dir, "memorization")
    os.makedirs(save_dir, exist_ok=True)

    match_indices = np.load(npy_path)

    # Load and sort generated image filenames
    gen_filenames = sorted([f for f in os.listdir(gen_im_dir) if f.endswith(".png")])
    generated_images = [Image.open(os.path.join(gen_im_dir, fname)).convert("RGB") for fname in gen_filenames]

    # Load training dataset
    train_dataset = CheXpertDataset(
        data_dir=train_data_dir,
        csv_name=csv_name,
        tokenizer=tokenizer,
        image_size=image_size
    )

    for i in range(min(num_to_plot, len(match_indices))):
        gen_img = generated_images[i]
        train_img_tensor = train_dataset[match_indices[i]]["pixel_values"]
        train_img = transforms.ToPILImage()(train_img_tensor)

        # Plot side by side
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(gen_img)
        axs[0].set_title("Generated")
        axs[0].axis("off")

        axs[1].imshow(train_img)
        axs[1].set_title("Most Similar Train")
        axs[1].axis("off")

        plt.tight_layout()

        # Save with the same name as generated image
        filename = gen_filenames[i]
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close()

    print(f"Saved {min(num_to_plot, len(match_indices))} match visualizations to {save_dir}")


def disease_occurence(train_data_dir, csv_name="train.csv"):
    """
    Count the number of occurrences of each disease in the training dataset.
    """
    import pandas as pd

    csv_path = os.path.join(train_data_dir, csv_name)
    df = pd.read_csv(csv_path)
    
    # self.df = self.df.iloc[:max_samples].reset_index(drop=True) # Limit the dataset size

    label_cols = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "Pleural Effusion", "Pleural Other",
        "Pneumonia", "Pneumothorax", "Support Devices", "No Finding"
    ]

    def get_label_combo(row):
        labels = [label for label in label_cols if row[label] == 1.0]
        return ", ".join(sorted(labels)) if labels else "no findings"

    df["combo"] = df.apply(get_label_combo, axis=1)

    # First 5000
    combo_counts_5k = df.iloc[:5000]["combo"].value_counts()

    # Full set
    combo_counts_full = df["combo"].value_counts()

    return combo_counts_5k, combo_counts_full

if __name__ == "__main__":
    args = parse_args()
    
    main()
    plot_matches(args.output_path, args.gen_im_dir, args.train_data_dir)
    
    
    # Get disease combo frequencies
    combo_counts_5k, combo_counts_full = disease_occurence(args.train_data_dir)

    # Convert to percentage
    percent_5k = 100 * combo_counts_5k / combo_counts_5k.sum()
    percent_full = 100 * combo_counts_full / combo_counts_full.sum()

    # Print top 30
    print("=== Top 10 Combinations in First 5000 ===")
    print(percent_5k.head(30).round(2).astype(str) + "%")
    # print("=== Top 10 Combinations in Full Dataset ===")
    # print(percent_full.head(30).round(2).astype(str) + "%")

