# PERNOSPHERE

![Logo](https://github.com/user-attachments/assets/bd6cbc46-9358-4b19-8eb1-8e291d3891e3)


## Vinegrape Leaf Disease Classifier

## Overview

PERNOSPHERE is a project made for the course AI-lab Computer Vision and NLP 2024/25. In this project, I implemented
a basic Convolutional Neural Network (CNN) to categorize the most common diseases in vinegrape plants.

---

## Author

Carlo Da Roma

## Features

* Basic PyTorch CNN
* Training with early stopping and methods to avoid overfitting
* Simple web Gradio interface
* Details and explanations about the diseases
* High accuracy test (0.95%+)
---

## Dataset

* The dataset contains images of vinegrape leaves, either diseased or healthy.
* Author: Rajarshi Mandal
* License: CC0 1.0 Universal

---

## Usage

### Setup

0. Install Gradio, OpenCV (`cv2`), PyTorch, Matplotlib, and Pandas

1. Download the dataset from the link below:
   [https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original]

2. MOVE ALL THE SUB-DIRECTORIES (test,train) OF THE DOWNLOADED ZIP INTO `Archive/trainTest/leaf`!

3. Create a usable dataset (CSV, images) by launching `support_function_test.py` and `support_function_train.py`

### Training

Run the training script:

```bash
python train_test.py
```

It will automatically stop.

### Inference

Launch the Gradio GUI to use the app:

```bash
python app.py
```

---

## Code Structure

* In the directory `Code` there are:

  * `model.py` – CNN definition
  * `dataloader.py` – Custom dataloader
  * `train_test.py` – Training and testing loop
  * `predict.py` – Predictor function
  * `GUI.py` – Gradio web app

* In the directory `Archive` there are the images of the dataset

* In the directory `Model` there is the saved binary model

* In the directory `Dataset` there are the `.csv` files and a support function to create a usable dataset

---

## Notes

* The model input images are resized to 128x128. The original size of the images is 512x512
* Early stopping patience is set to 3 epochs by default
* Typically, training for 9 to 10 epochs is enough to reach a test accuracy of at least 95%

### IMPORTANT!

* To achieve the best results, take the photo of the leaf on a sheet of paper.

---
