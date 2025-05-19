'''

model.py

'''

import torch
from torch.utils.data import DataLoader
from torch import nn
import torchmetrics
from dataloader import MyDataloader
import random
import pandas as pd
import matplotlib.pyplot as plt

class OurCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2), 
        )
        self.mlp = nn.Sequential(
            nn.Linear(9216, 512),
            nn.ReLU(),
            nn.Dropout(0.5), 
        
            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x