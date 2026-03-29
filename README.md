# Retinal OCT Classification using EfficientNet + CBAM

## 🔍 Overview
This project classifies OCT images into:
- CNV
- DME
- DRUSEN
- NORMAL

## 🚀 Features
- EfficientNet + CBAM feature extraction
- Random Forest classifier
- GradCAM visualization
- Confidence scores
- FastAPI deployment

## 📊 Results
- Accuracy: ~97%

## 🧠 Tech Stack
- PyTorch
- FastAPI
- Scikit-learn
- OpenCV

## 📡 API Endpoint
`/predict`

Upload an image → get prediction + GradCAM