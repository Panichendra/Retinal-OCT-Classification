---
title: Retinal OCT Classification
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# 🧠 Retinal OCT Classification System

This project is an AI-based web application that classifies retinal OCT images.

...

## 🚀 What This Project Does

Upload an OCT image and the system will:

- 🔍 Predict the disease (CNV, DME, DRUSEN, NORMAL)
- 📊 Show confidence score
- 📌 Show top-2 possible predictions
- 🔥 Highlight important regions using Grad-CAM
- 📖 Provide a short explanation of the disease

---

## 🧠 Model Used

- EfficientNet-B0 (Deep Learning model)
- CBAM (Attention mechanism to improve focus)
- Random Forest (for final classification)
- Grad-CAM (for visualization)

---

## 📂 Classes

The model classifies images into:

- **CNV** – Abnormal blood vessel growth
- **DME** – Fluid accumulation due to diabetes
- **DRUSEN** – Deposits under the retina
- **NORMAL** – No disease detected

---

## 🖥️ How to Use the App

1. Upload an OCT image  
2. Click **"Analyze"**  
3. View the results:

   - Prediction  
   - Confidence  
   - Top-2 predictions  
   - Disease explanation  
   - Grad-CAM visualization  

---

## ⚠️ Important Note

This tool is for **educational and research purposes only**.

👉 It is **NOT a medical diagnosis tool**  
👉 Always consult a doctor for medical decisions  

---



## 🛠️ Technologies Used

- Python
- PyTorch
- Scikit-learn
- OpenCV
- Gradio (UI)
- Hugging Face Spaces (Deployment)

---

## 💼 Project Highlights

- Hybrid model (Deep Learning + Machine Learning)
- Explainable AI using Grad-CAM
- Real-time web application
- Clean and interactive UI

---

## 👨‍💻 Author

**PaniChendra**

---

