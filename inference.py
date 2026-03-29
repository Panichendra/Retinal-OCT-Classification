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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# HUGGINGFACE LINKS (FIXED)
# =========================

MODEL_URL = "https://huggingface.co/PaniChendra/retinal-oct-model/resolve/main/Backend/MyProject_effcbam_model.pth"
RF_URL = "https://huggingface.co/PaniChendra/retinal-oct-model/resolve/main/Backend/MyProject_rf_model.pkl"
CLASSES_URL = "https://huggingface.co/PaniChendra/retinal-oct-model/resolve/main/Backend/MyProject_classes.json"

# =========================
# DOWNLOAD FUNCTION
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
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max,_ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max], dim=1)
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
        x = torch.flatten(x, 1)
        return self.fc(x)


# =========================
# LAZY LOADING (IMPORTANT)
# =========================

model = None
rf = None
class_names = None
original_fc = None

def load_models():
    global model, rf, class_names, original_fc

    if model is None:
        print("Loading models...")

        download_file(MODEL_URL, "MyProject_effcbam_model.pth")
        download_file(RF_URL, "MyProject_rf_model.pkl")
        download_file(CLASSES_URL, "MyProject_classes.json")

        model = EfficientNet_CBAM(4).to(device)
        model.load_state_dict(torch.load("MyProject_effcbam_model.pth", map_location=device))
        model.eval()

        original_fc = model.fc

        rf = joblib.load("MyProject_rf_model.pkl")

        with open("MyProject_classes.json") as f:
            class_names = json.load(f)

        print("Models loaded successfully")


# =========================
# TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# =========================
# GRADCAM
# =========================

gradients = None
activations = None

def save_gradient(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def save_activation(module, input, output):
    global activations
    activations = output


def register_hooks():
    target_layer = model.features[-1]
    target_layer.register_forward_hook(save_activation)
    target_layer.register_backward_hook(save_gradient)


def generate_gradcam(img_tensor, class_idx):
    model.zero_grad()
    output = model(img_tensor)
    loss = output[:, class_idx]
    loss.backward()

    grads = gradients
    acts = activations

    pooled_grads = torch.mean(grads, dim=[0,2,3])

    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(acts, dim=1).squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap


# =========================
# PREDICT
# =========================

def predict(image_path):

    load_models()
    register_hooks()

    img = Image.open(image_path).convert("RGB")
    original = np.array(img)

    img_tensor = transform(img).unsqueeze(0).to(device)

    # FEATURE EXTRACTION
    model.fc = nn.Identity()
    with torch.no_grad():
        features = model(img_tensor).cpu().numpy()

    probs = rf.predict_proba(features)[0]

    pred_idx = np.argmax(probs)
    prediction = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    # TOP-2
    top2_idx = probs.argsort()[-2:][::-1]
    top2 = [
        {"class": class_names[i], "confidence": float(probs[i])}
        for i in top2_idx
    ]

    # GRADCAM
    model.fc = original_fc
    output = model(img_tensor)
    class_idx = torch.argmax(output, dim=1).item()

    heatmap = generate_gradcam(img_tensor, class_idx)
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    gradcam_path = "gradcam_output.jpg"
    cv2.imwrite(gradcam_path, overlay)

    return prediction, confidence, top2, gradcam_path
