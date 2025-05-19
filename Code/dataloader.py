'''

dataloader.py

This file contains the function that allows you to load data from the Archive.

'''

from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os
from PIL import Image

class MyDataloader(Dataset):
    def __init__(self, labels_csv, img_dir, transform=None, seed=0, resize=(128, 128)):
        self.img_dir = img_dir
        data = pd.read_csv(labels_csv).sample(frac=1, random_state=seed).reset_index(drop=True)
        self.labels = data

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
