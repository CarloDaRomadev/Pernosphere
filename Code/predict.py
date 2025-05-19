'''

predict.py

This file contains the function that predict a label for a photo using the model.

'''

import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from model import OurCNN  
import numpy as np

dinfo = {
    "Black Rot": {
        "description": "Grape black rot is a fungal disease caused by an ascomycetous fungus, Guignardia bidwellii...",
        "wiki": "https://en.wikipedia.org/wiki/Black_rot_(grape_disease)"
    },
    "ESCA": {
        "description": "Esca is a grape disease of mature grapevines...",
        "wiki": "https://en.wikipedia.org/wiki/Esca_(disease)"
    },
    "Healty": {
        "description": "No signs of disease. Leaf appears green, normal and HEALTY.",
        "wiki": "https://en.wikipedia.org/wiki/Vineyard"
    },
    "Leaf Blight": {
        "description": "Causes irregular brown and yellow lesions on leaves, often leading to defoliation.",
        "wiki": "https://en.wikipedia.org/wiki/Leaf_blight"
    }
}

try:
    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    elif hasattr(torch.version, "hip") and torch.version.hip:
        dev = "hip"
    else:
        dev = "cpu"
except AttributeError:
    dev = "cpu"

net = OurCNN().to(dev)
net.load_state_dict(torch.load('AI-Lab_project/Model/model_leaf.pth', map_location=dev))
net.eval() 


tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

labels = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']

def predicter(image: Image.Image):
    img_t = tfm(image).unsqueeze(0).to(dev)

    with torch.no_grad():
        outputs = net(img_t)
        probs = torch.softmax(outputs, dim=1)
        
        conf, pred = torch.max(probs, dim=1)
        confidence = conf.item()
        
        if confidence > 0.50:
            predicted_class = labels[pred.item()]
            return f"{predicted_class} ({confidence * 100:.2f}%)"
        else:
            return "Error: The uploaded image is not confidently recognized as a grapevine leaf."
