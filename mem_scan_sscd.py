import torch
from sscd_model import Model

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

    parser.add_argument("--train_data_dir", type=str, required=True, help="Folder containing the training data.")
    parser.add_argument("--gen_im_dir", type=str, required=True, help="Folder containing generated images.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for loading training data.")
    parser.add_argument("--image_size", type=int, default=512, help="Image size (assumed square) for resizing and comparison.")
    parser.add_argument("--output_path", type=str, default="best_train_matches.npz", help="Where to save the best match indices.")


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
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Model("TV_RESNET50", 512, 3.0)
        weights = torch.load("sscd_disc_mixup.torchvision.pt")
        model.load_state_dict(weights)
        model.eval()
        model.to(device)


        # Load and find features for all generated images
        generated_images = load_generated_images(args.gen_im_dir, device)
        gen_tensor = torch.stack(generated_images)
        gen_loader = DataLoader(gen_tensor, batch_size=args.batch_size, shuffle=False)
        gen_feats = []
        for batch in tqdm(gen_loader, desc="Encoding generated images"):
            batch = batch.to(device) # [B(4), C(3), H(512), W(512)]
            features = model(batch)
            gen_feats.append(features.view(features.shape[0], -1)) # [B, D(C*H*W)]
        gen_feats = torch.cat(gen_feats, dim=0)  # [N, D]
        N = gen_feats.size(0)
        print(f"Stored features for {N} generated images.")

        # Load training dataset
        train_dataset = CheXpertDataset(
            data_dir=args.train_data_dir,
            csv_name="train.csv",
            tokenizer=tokenizer,
            image_size=args.image_size
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

        # Initialize best scores, indices and current index (necessary cause working in batches)
        best_scores = torch.full((N,), -float("inf"), device=device)
        best_indices = torch.full((N,), -1, dtype=torch.long, device=device)
        current_index = 0
        for batch in tqdm(train_loader, desc="Cosine similarity between gen and train features"):
            train_imgs = batch["pixel_values"].to(device,)
            train_feats = model(train_imgs).view(train_imgs.size(0), -1)

            # Compute [N, B] similarity
            sim = torch.matmul(gen_feats, train_feats.T)

            # For each generated image, update best match if current sim is higher
            max_vals, max_idxs = sim.max(dim=1)
            update_mask = max_vals > best_scores
            best_scores[update_mask] = max_vals[update_mask]
            best_indices[update_mask] = current_index + max_idxs[update_mask]

            current_index += train_imgs.size(0)

        # Save results
        np.savez(args.output_path, indices=best_indices.cpu().numpy(), scores=best_scores.cpu().numpy())
        print(f"Saved best match indices and scores to {args.output_path}")


def plot_matches(npy_path, gen_im_dir, train_data_dir, csv_name="train.csv", image_size=512, num_to_plot=1000):
    """
    Save side-by-side plots of generated images and their best matching training images.
    """

    import os

    # Output folder for memorization matches
    save_dir = os.path.join(gen_im_dir, "memorization")
    os.makedirs(save_dir, exist_ok=True)

    data = np.load(npy_path)
    print(data)
    match_indices = data["indices"]
    similarity_scores = data["scores"]


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
        axs[1].set_title(f"Train (sim={similarity_scores[i]:.3f})")
        axs[1].axis("off")

        plt.tight_layout()

        # Save with the same name as generated image
        filename = gen_filenames[i]
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close()

    print(f"Saved {min(num_to_plot, len(match_indices))} match visualizations to {save_dir}")

if __name__ == "__main__":
    args = parse_args()
    
    main()
    plot_matches(args.output_path, args.gen_im_dir, args.train_data_dir)

    # Show prompts for the first 10 training samples
    print("\nPrompts for the first 10 training samples in order:")

    train_dataset = CheXpertDataset(
        data_dir=args.train_data_dir,
        csv_name="train.csv",
        tokenizer=tokenizer,
        image_size=args.image_size
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    for idx, batch in enumerate(islice(train_loader, 10)):
        print(f"Sample {idx}")
        print()

    print("\nNote: These are the samples that were repeated during training.")
