# 🏥 Robust Zero-Watermarking of Medical Images using Deep CNN

A deep learning–based framework for **secure copyright protection and authentication of medical images** using **zero-watermarking and CNN feature extraction**. This system ensures **data integrity without modifying the original image**, making it ideal for healthcare applications.

---

## 🚀 Overview

With the rapid digitization of healthcare, medical images (MRI, CT, X-ray) are vulnerable to **tampering, duplication, and unauthorized distribution**. Traditional watermarking methods modify image content, which is not suitable for medical use.

This project introduces a **CNN-based zero-watermarking system** that:
- Extracts **deep, discriminative features**
- Generates a **secure watermark signature (without embedding)**
- Enables **robust verification even after attacks**

---

## 🧠 Key Features

- 🔍 Deep CNN Feature Extraction (VGG16 / ResNet / MobileNet)
- 🔐 Zero-Watermarking (No Image Modification)
- 🛡️ Robust Against Attacks (Noise, Compression, Rotation, Scaling)
- 📊 Performance Metrics: PSNR, SSIM, Correlation Coefficient
- ⚡ Tamper Detection & Ownership Verification
- 🏥 Healthcare Ready (PACS, Cloud, Telemedicine Integration)

---

## 🏗️ System Architecture

The system follows a modular pipeline:

1. Image Acquisition – Load medical images  
2. Preprocessing – Normalize, resize, enhance quality  
3. Feature Extraction – CNN extracts deep features  
4. Zero-Watermark Generation – Create unique watermark signature  
5. Watermark Verification – Compare features for authenticity  
6. Attack Simulation – Apply distortions for robustness testing  
7. Performance Evaluation – Analyze PSNR, SSIM, CC  

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries:** NumPy, OpenCV, TensorFlow/Keras, Matplotlib, Scikit-learn  
- **Model:** Pretrained CNN (VGG16 / ResNet / MobileNet)  
- **Environment:** Jupyter Notebook / Spyder  

---

## 📂 Project Structure 
```plaintext 
├── data/                     # Medical image dataset 
├── preprocessing/           # Image cleaning & normalization 
├── feature_extraction/      # CNN model implementation 
├── watermarking/            # Zero-watermark generation 
├── verification/            # Authentication logic 
├── attack_simulation/       # Noise, rotation, compression tests 
├── results/                 # Evaluation metrics & outputs 
├── notebooks/               # Jupyter notebooks 
└── README.md 
```

## 📊 Performance Evaluation

The model is evaluated using:

- **PSNR (Peak Signal-to-Noise Ratio)** → Image quality  
- **SSIM (Structural Similarity Index)** → Structural integrity  
- **Correlation Coefficient (CC)** → Watermark accuracy  

✔ High robustness observed under multiple attack scenarios.

---

## 🔬 How It Works

1. Extract deep features from original image using CNN  
2. Generate a unique watermark signature  
3. Store watermark securely (not embedded in image)  
4. For verification:
   - Extract features from test image  
   - Compare with stored watermark  
   - Output: **Verified / Tampered**

---

## 🖥️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git

# Navigate to project directory
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py
```
## 📌 Applications

-   🏥 Hospital Image Security (PACS)
    
-   ☁️ Cloud Healthcare Systems
    
-   📡 Telemedicine Platforms
    
-   🔐 Medical Data Copyright Protection
    

* * *

## ⚠️ Limitations

-   Requires high-quality training data
    
-   Performance depends on CNN model selection
    
-   Computational cost for deep feature extraction
    

* * *

## 🔮 Future Enhancements

-   Integration with **Blockchain for secure storage**
    
-   Real-time deployment in hospital systems
    
-   Lightweight models for faster inference
    
-   Multi-modal medical image support
    

* * *

## 🤝 Contributing

Contributions are welcome! Feel free to:

-   Fork the repo
    
-   Create a feature branch
    
-   Submit a pull request
    

* * *

## 📜 License

This project is for academic and research purposes.

* * *



