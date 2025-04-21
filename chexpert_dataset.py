import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CheXpertDataset(Dataset):
    def __init__(self, data_dir, tokenizer, csv_name="train.csv", image_size=512, max_samples=5000, overfit_n=None, overfit_repeat=None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer

        csv_path = os.path.join(data_dir, csv_name)
        self.df = pd.read_csv(csv_path)
        self.df = self.df.iloc[:max_samples].reset_index(drop=True) # Limit the dataset size

        self.overfit_n = overfit_n
        self.overfit_repeat = overfit_repeat

        self.label_cols = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
            "Lung Opacity", "Pleural Effusion", "Pleural Other",
            "Pneumonia", "Pneumothorax", "Support Devices", "No Finding"
        ]
        self.df[self.label_cols] = self.df[self.label_cols].fillna(0.0)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        if self.overfit_n is not None and self.overfit_repeat is not None:
            self._create_overfit_mapping()

    def _create_overfit_mapping(self):
        total_samples = len(self.df)
        self.mapping = []

        # repeat the first overfit_n samples overfit_repeat times
        for i in range(self.overfit_n):
            self.mapping.extend([i] * self.overfit_repeat)

        # keep the rest of the samples
        for i in range(self.overfit_n, total_samples):
            self.mapping.append(i)

    def __len__(self):
        return len(self.mapping) if hasattr(self, 'mapping') else len(self.df)

    def __getitem__(self, idx):
        if hasattr(self, 'mapping'):
            idx = self.mapping[idx]
        row = self.df.iloc[idx]

        relative_path = row["Path"].replace("CheXpert-v1.0-small/", "")
        img_path = os.path.join(self.data_dir, relative_path)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image) # Don't care about the interpolation method of transforming image
        
        # Generate text prompt
        if random.random() < 0.1:
            prompt = ""
        else:
            present_labels = [label for label in self.label_cols if row[label] == 1.0]
            disease_text = ", ".join(present_labels) if present_labels else "no findings"
            # Support devices
            has_devices = row["Support Devices"] == 1.0
            device_str = "with support devices" if has_devices else "without support devices"
            # View type
            view_type = str(row.get("Frontal/Lateral", "frontal")).strip().lower()
            if view_type not in ["frontal", "lateral"]:
                view_type = "frontal"
                
            prompt = f"A {view_type} view of a chest X-ray with {disease_text} {device_str}"

        # Tokenize prompt
        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids[0]

        return {
            "pixel_values": image,
            "input_ids": input_ids
        }
