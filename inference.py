import torch
import torch.nn as nn
import numpy as np
import joblib
import json
import cv2
import os
import urllib.request

from PIL import Image
from torchvision import transforms, models

device = torch.device("cpu")

# =========================
# HUGGINGFACE LINKS
# =========================

MODEL_URL = "https://huggingface.co/PaniChendra/retinal-oct-model/resolve/main/Backend/MyProject_effcbam_model.pth"
RF_URL = "https://huggingface.co/PaniChendra/retinal-oct-model/resolve/main/Backend/MyProject_rf_model.pkl"
CLASSES_URL = "https://huggingface.co/PaniChendra/retinal-oct-model/resolve/main/Backend/MyProject_classes.json"

# =========================
# DOWNLOAD
# =========================

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

# =========================
# CBAM
# =========================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2,1,7,padding=3,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max,_ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg,max],dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x * self.sa(x * self.ca(x))

# =========================
# MODEL
# =========================

class EfficientNet_CBAM(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        self.features = base.features
        self.cbam = CBAM(1280)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        return self.fc(x)

# =========================
# GLOBALS
# =========================

model = None
rf = None
class_names = None
original_fc = None

# =========================
# LOAD MODELS (RUN ONCE)
# =========================

def load_models():
    global model, rf, class_names, original_fc

    if model is None:
        download_file(MODEL_URL, "model.pth")
        download_file(RF_URL, "rf.pkl")
        download_file(CLASSES_URL, "classes.json")

        model = EfficientNet_CBAM(4).to(device)
        model.load_state_dict(torch.load("model.pth", map_location=device))
        model.eval()

        original_fc = model.fc

        rf = joblib.load("rf.pkl")

        with open("classes.json") as f:
            class_names = json.load(f)

        print("✅ Models loaded successfully")

# =========================
# TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485]*3,[0.229]*3)
])

# =========================
# PREDICTION ONLY
# =========================

def predict_only(image_path):
    global model

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.fc = nn.Identity()

    with torch.no_grad():
        features = model(img_tensor).cpu().numpy()

    probs = rf.predict_proba(features)[0]

    pred_idx = np.argmax(probs)

    prediction = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    top2_idx = probs.argsort()[-2:][::-1]
    top2 = [
        {"class": class_names[i], "confidence": float(probs[i])}
        for i in top2_idx
    ]

    return prediction, confidence, top2, pred_idx, img_tensor, img

# =========================
# GRADCAM
# =========================

def generate_gradcam_only(pred_idx, img_tensor, original_img):

    global model, original_fc

    model.fc = original_fc

    img_tensor.requires_grad = True
    output = model(img_tensor)

    loss = output[:, pred_idx]
    loss.backward()

    gradients = img_tensor.grad
    heatmap = gradients.mean(dim=1).squeeze().cpu().numpy()

    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap) + 1e-8

    original = np.array(original_img)

    heatmap = cv2.resize(heatmap,(original.shape[1],original.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original,0.6,heatmap,0.4,0)

    path = "gradcam_output.jpg"
    cv2.imwrite(path, overlay)

    return path
