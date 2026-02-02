# Intelligent Plant Disease Detection
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Jupyter Notebook](https://img.shields.io/badge/Tools-Jupyter%20Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

A Deep Learning project designed to detect and classify diseases in tomato plants using Convolutional Neural Networks (CNN). This system processes images of plant leaves to identify health status and various specific conditions, aiding in early diagnosis and crop management.

## 📌 Project Overview

This repository contains a Jupyter Notebook that implements an end-to-end pipeline for image classification. It utilizes **TensorFlow** and **Keras** to build a CNN model capable of recognizing 10 distinct classes of tomato leaf conditions, including healthy leaves and various bacterial, fungal, and viral diseases.

The workflow includes:

1. **Data Ingestion:** Loading images from a structured directory using **OpenCV**.
2. **Preprocessing:** Image resizing, normalization, and label encoding.
3. **Data Augmentation:** Applying random flips, rotations, and zooms to improve model generalization.
4. **Model Training:** Training a Sequential CNN with custom architecture.
5. **Evaluation:** Analyzing performance using Confusion Matrices and Classification Reports.

## 🚀 Features

* **Multi-Class Classification:** Capable of distinguishing between 10 different categories of tomato plant health.
* **Image Preprocessing Pipeline:** Automated resizing (100x100 pixels) and pixel intensity normalization.
* **Integrated Data Augmentation:** Built-in layers for random contrast, flip, rotation, and zoom to reduce overfitting.
* **Custom CNN Architecture:** Utilizes multiple Convolutional layers, Max Pooling, and Global Average Pooling for efficient feature extraction.
* **Performance Metrics:** Includes code for visualizing accuracy and generating detailed classification reports.

## 🌿 Target Classes

The model is trained to detect the following classes:

1. Tomato - Bacterial Spot
2. Tomato - Early Blight
3. Tomato - Late Blight
4. Tomato - Leaf Mold
5. Tomato - Septoria Leaf Spot
6. Tomato - Spider Mites (Two-spotted spider mite)
7. Tomato - Target Spot
8. Tomato - Tomato Mosaic Virus
9. Tomato - Tomato Yellow Leaf Curl Virus
10. Tomato - Healthy

## 🛠️ Installation & Prerequisites

To run this project locally, ensure you have **Python 3.x** installed. You will need the following libraries:

```bash
pip install pandas numpy matplotlib tensorflow opencv-python

```

* **TensorFlow/Keras:** For building and training the neural network.
* **OpenCV (cv2):** For image reading and processing.
* **Pandas & NumPy:** For data manipulation and matrix operations.
* **Matplotlib:** For visualization.

## 📂 Project Structure

```
├── Intelligent Plant Disease Detection.ipynb   # Main project notebook
├── README.md                                   # Project documentation
└── [Dataset Folder]                            # (External) Source images folder

```

## 💻 Usage

1. **Clone the Repository:**
```bash
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection

```


2. **Prepare the Dataset:**
Ensure your dataset is organized in folders by class name (e.g., `.../tomato/train/Tomato___healthy`).
*Update the `path` variable in the notebook to point to your local dataset directory.*
3. **Run the Notebook:**
Open the Jupyter Notebook and execute the cells sequentially.
```bash
jupyter notebook "Intelligent Plant Disease Detection.ipynb"

```


4. **Training:**
The notebook will load the images, train the model, and display the training progress (loss/accuracy).
5. **Prediction:**
Use the provided prediction code block to classify new leaf images.

## 🧠 Model Architecture

The project utilizes a `Sequential` model structure:

* **Input & Augmentation:** Random transformations for robustness.
* **Convolutional Blocks:** Multiple `Conv2D` layers (120 and 64 filters) with `ReLU` activation for feature extraction, paired with `MaxPooling2D`.
* **Pooling:** `GlobalAveragePooling2D` to reduce dimensionality.
* **Dense Layers:** Fully connected layers for classification logic.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improving the model architecture, adding new plant types, or optimizing the preprocessing pipeline, please feel free to fork the repository and submit a pull request.

## 📄 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).
