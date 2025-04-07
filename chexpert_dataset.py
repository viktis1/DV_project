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
        
        if random.random() < 0.1: # Train unconditionally 10% of the time
            prompt = ""  
        else:
            # Extract present labels
            present_labels = [label for label in self.label_cols if row[label] == 1.0]

            # Decide how to phrase the prompt
            if "Support Devices" in present_labels:
                patho_labels = [label for label in present_labels if label != "Support Devices"]
                if patho_labels: # disease and support devices
                    disease_text = ", ".join(patho_labels)
                    base_prompt = random.choice(self.cond_prompts).format(disease_text)
                    prompt = base_prompt + ", with visible support devices"
                else: # No diseases and only support diseases
                    prompt = random.choice(self.device_prompts)
            elif present_labels:
                disease_text = ", ".join(present_labels)
                prompt = random.choice(self.cond_prompts).format(disease_text)
            else:
                prompt = random.choice(self.generic_prompts)


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
