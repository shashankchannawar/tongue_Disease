# AI-powered Tongue Disease Detection using CNNs

## Overview
This project utilizes a Convolutional Neural Network (CNN) based deep learning model to detect oral diseases from tongue images. It aims to assist in the early detection of oral infections, deficiencies, and potential cancer indicators using Medical AI.

## Features
- **Disease Detection**: Classifies tongue images into various categories (e.g., Healthy, Infection, etc.).
- **Deep Learning**: Uses TensorFlow/Keras for CNN implementation.
- **Data Augmentation**: Enhances robustness using image augmentation techniques.
- **Transfer Learning**: Supports pretrained models (e.g., MobileNetV2, ResNet) for better accuracy.

## Tech Stack
- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy / Pandas**
- **Flask** (for the web interface)

## Dataset
Please place your dataset in the `data/raw` directory organized by class folders:
```
data/
  raw/
    healthy/
    cancer/
    infection/
```

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python src/train.py
   ```
3. Run the web interface:
   ```bash
   python web_app/app.py
   ```
