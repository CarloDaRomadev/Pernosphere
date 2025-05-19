## 📌 Overview  
PERNOSPHERE is a **Computer Vision** project developed for the AI-lab Computer Vision and NLP course (2024/25). It uses a **PyTorch-based CNN** to classify common diseases in vinegrape leaves with **>95% test accuracy**.  

Key Features:  
✅ Basic CNN architecture  
✅ Early stopping & overfitting prevention  
✅ Gradio web interface for easy inference  
✅ Detailed disease information  

---

## 🧑‍💻 Author  
**Carlo Da Roma**  

---

## 📂 Dataset  
- **Source**: [Kaggle - Grape Disease Dataset](https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original) (by Rajarshi Mandal, CC0 1.0 License)  
- **Contents**: Images of healthy/diseased vinegrape leaves (original size: 512x512, resized to 128x128 for model input).  

---

## 🚀 Usage  

### ⚙️ Setup  
1. **Install dependencies**:  
   ```bash
   pip install gradio torch torchvision opencv-python matplotlib pandas
   ```  
2. **Download dataset** from the Kaggle link above.  
3. **Organize files**: Move all subdirectories (`test`, `train`) into `Archive/trainTest/leaf/`.  
4. **Preprocess data**: Run these to generate CSVs:  
   ```bash
   python support_function_train.py
   python support_function_test.py
   ```  

### 🏋️ Training  
```bash
python train_test.py
```  
- Training auto-stops after 3 epochs without improvement (adjustable in code).  
- Typically achieves **95%+ accuracy in 9-10 epochs**.  

### 🌐 Web App  
Launch the Gradio interface:  
```bash
python app.py
```  
**Tip**: For best results, photograph leaves on a plain white background.  

---

## 🗂️ Code Structure  
```  
PERNOSPHERE/  
├── Code/  
│   ├── model.py           # CNN architecture  
│   ├── dataloader.py      # Custom data loader  
│   ├── train_test.py      # Training/testing loops  
│   ├── predict.py         # Inference functions  
│   └── GUI.py             # Gradio app  
├── Archive/               # Raw dataset (train/test images)  
├── Model/                 # Saved model binaries  
└── Dataset/               # Processed CSVs & support scripts  
```  

---

## 📝 Notes  
- **Image Resolution**: Model uses 128x128 inputs (downscaled from 512x512).  
- **Early Stopping**: Default patience=3 epochs.  
- **Performance**: Achieves 95%+ test accuracy consistently.  

--- 

🛠️ **Contribution & Issues**  
Feel free to open issues or suggest improvements!  

---  
