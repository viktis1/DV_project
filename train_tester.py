from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from chexpert_dataset import CheXpertDataset
import time

def time_the_dataloader(): # This took 12 min to run. I am ok with that
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    dataset = CheXpertDataset(
        data_dir="C:/Users/isaks/.cache/kagglehub/datasets/ashery/chexpert/versions/1",
        csv_name="train.csv",
        tokenizer=tokenizer
    )
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    start = time.time()
    for batch in train_loader:
        pass
    print("Full epoch time:", time.time() - start)


def main():
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    dataset = CheXpertDataset(
        data_dir="/dtu/blackhole/1d/214141/CheXpert-v1.0-small",
        csv_name="train.csv",
        tokenizer=tokenizer
    )
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    for batch in train_loader:
        print("pixel_values:", batch["pixel_values"].shape)  # torch.Size([B, 3, 512, 512])
        print("input_ids:", batch["input_ids"].shape)        # torch.Size([B, embed_dim])
        break


if __name__ == "__main__":
    main()
    time_the_dataloader()
