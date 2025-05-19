'''
visualizer.py (test function)

Description:

This script loads a custom dataset of grapevine 
leaf images, randomly selects 16 samples, and visualizes 
them in a 4x4 grid using matplotlib. Each image is 
displayed along with its corresponding disease label.

'''
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from dataloader import MyDataloader

label_map = {
    'Black-Rot': 0, 'ESCA': 1, 'NO (Healthy)': 2, 'Leaf-blight': 3
}

data = MyDataloader("AI-Lab_project/Dataset/dataset_leaf_labeled_train.csv", "AI-Lab_project/Archive/all_leaf_train", transform=None)
figure = plt.figure(figsize=(15, 15)) 
cols, rows = 4, 4 
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(data), size=(1,)).item()
    image, label = data[sample_idx]
    disease = [name for name, number in label_map.items() if number == label][0]
    figure.add_subplot(rows, cols, i)
    plt.title(f"DISEASE: {disease}")
    plt.imshow(image.permute(1, 2, 0)) # Here I swap the channels of the image to have RGB instead of GBR...
    plt.axis('off')

plt.show()