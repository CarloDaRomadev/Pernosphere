# PERNOSPHERE: Vinegrape Leaf Disease Classifier  

## 📝 Overview  
PERNOSPHERE is a project developed for the **AI-lab Computer Vision and NLP 2024/25** course. It implements a **Convolutional Neural Network (CNN)** using PyTorch to classify common diseases in vinegrape leaves.  

---

## 👨‍💻 Author  
**Carlo Da Roma**  

---

## ✨ Features  
- Basic PyTorch CNN implementation  
- Training with early stopping and overfitting prevention  
- Simple Gradio web interface  
- Detailed disease information  
- High test accuracy (95%+)  

---

## 📊 Dataset  
- Contains images of healthy and diseased vinegrape leaves  
- **Author**: Rajarshi Mandal  
- **License**: CC0 1.0 Universal  
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original)  

---

## 🛠️ Usage  

### ⚙️ Setup  
1. Install required packages:  
   ```bash
   pip install gradio opencv-python torch matplotlib pandas
   ```  
2. Download the dataset from the Kaggle link above  
3. Move all sub-directories (`test`, `train`) into `Archive/trainTest/leaf/`  
4. Generate CSV files by running:  
   ```bash
   python support_function_train.py
   python support_function_test.py
   ```  

### 🏋️ Training  
Run the training script:  
```bash
python train_test.py
```  
*Automatically stops when validation accuracy doesn't improve for 3 consecutive epochs.*  

### 🌐 Web Interface  
Launch the Gradio app:  
```bash
python app.py
```  

---

## 📂 Code Structure  
```
PERNOSPHERE/
├── Code/
│   ├── model.py          # CNN architecture
│   ├── dataloader.py     # Custom data loader
│   ├── train_test.py     # Training/testing script
│   ├── predict.py        # Prediction functions
│   └── GUI.py            # Gradio interface
├── Archive/              # Raw dataset images
├── Model/                # Saved model files
└── Dataset/              # Processed CSV files
```

---

## ℹ️ Notes  
- Input images are resized to **128x128** (original: 512x512)  
- Default early stopping patience: **3 epochs**  
- Typically achieves **>95% accuracy** in 9-10 epochs  

### ⚠️ IMPORTANT  
For best results, photograph leaves on a plain white background (e.g., sheet of paper).  

--- 

*(Note: This README maintains all original content while improving GitHub formatting with headers, lists, and code blocks for better readability.)*
