import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CheXpertDataset(Dataset):
    def __init__(self, data_dir, tokenizer, csv_name="train.csv", image_size=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer

        csv_path = os.path.join(data_dir, csv_name)
        self.df = pd.read_csv(csv_path)

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

        # Prompt templates
        self.cond_prompts = [
            "An X-ray image of a chest with {}",
            "An X-ray image showing signs of {} in the lungs",
            "A radiograph of a patient with {}",
            "A chest radiograph demonstrating {}",
            "A chest X-ray showing {}",
            "A chest X-ray with signs of {}",
            "An X-ray scan showing {} in the thoracic cavity",
            "A medical X-ray of the lungs exhibiting {}",
            "A diagnostic chest X-ray with {}",
            "A chest radiograph indicating {}",
            "A lung with {} seen in a chest X-ray image",
            "An X-ray of lungs affected by {}",
            "A chest X-ray indicating {}",
            "An X-ray showing pathological features of {} in the lungs",
            "A patient’s chest radiograph demonstrating signs of {}",
        ]

        self.generic_prompts = [
            "A chest X-ray image",
            "A frontal view of a chest X-ray",
            "A chest radiograph",
            "A medical X-ray of the lungs",
            "A standard chest X-ray of lungs",
            "An X-ray scan of the thorax",
            "A diagnostic X-ray image of the chest",
            "An X-ray photo of human lungs",
            "An X-ray image of a patient’s chest",
            "A radiograph showing normal lung structure",
            "A lung X-ray taken in clinical setting",
            "A medical radiographic image of the upper torso",
            "An X-ray of the thoracic cavity",
            "A clear chest X-ray",
        ]

        self.device_prompts = [ # This is only used if no diseases, but there are support devices
            "A chest X-ray showing visible support devices",
            "An X-ray image of the chest with medical equipment present",
            "A radiograph demonstrating the presence of support devices",
            "A chest X-ray revealing tubes or other medical devices",
            "An X-ray image of a patient with inserted support devices",
            "A diagnostic chest radiograph showing support hardware",
            "A frontal chest X-ray displaying medical support devices",
            "An X-ray image with visible catheters or tubes",
            "A chest radiograph including support devices",
            "A clinical X-ray indicating presence of external medical devices"
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
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

            # --- Metadata - we also condition on this (90% we use) ---
            if random.random() < 0.9:
                sex = row.get("Sex", "Unknown")
                age = int(row.get("Age", -1)) if not pd.isna(row.get("Age")) else "Unknown"
                view_type = row.get("Frontal/Lateral", "").strip()
                projection = str(row.get("AP/PA", "")).strip()

                # Build metadata string
                meta_parts = []
                view_type = str(row.get("Frontal/Lateral", "")).strip()
                if view_type and view_type.lower() != "nan":
                    meta_parts.append(f"{view_type.lower()} view")

                projection = str(row.get("AP/PA", "")).strip()
                if projection and projection.lower() != "nan":
                    meta_parts.append(f"{projection} projection")

                sex = str(row.get("Sex", "")).strip()
                if sex and sex.lower() != "nan":
                    meta_parts.append(f"{sex.lower()} patient")

                age = int(row.get("Age", -1)) if not pd.isna(row.get("Age")) else "Unknown"
                if not pd.isna(age):
                    age = int(age)
                    meta_parts.append(f"{age}-year-old")

                meta_parts.append(", ")
                meta_str = ", ".join(meta_parts)
            else:
                meta_str = ""

            # --- Prompt templates ---
            if "Support Devices" in present_labels:
                patho_labels = [label for label in present_labels if label != "Support Devices"]
                if patho_labels:
                    disease_text = ", ".join(patho_labels)
                    base_prompt = random.choice(self.cond_prompts).format(disease_text)
                    prompt = f"{meta_str}{base_prompt}, with visible support devices"
                else:
                    device_prompt = random.choice(self.device_prompts)
                    prompt = f"{meta_str}{device_prompt}"
            elif present_labels:
                disease_text = ", ".join(present_labels)
                base_prompt = random.choice(self.cond_prompts).format(disease_text)
                prompt = f"{meta_str}{base_prompt}"
            else:
                generic_prompt = random.choice(self.generic_prompts)
                prompt = f"{meta_str}{generic_prompt}"

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
