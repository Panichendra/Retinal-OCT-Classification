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

# ================= SETTINGS =================
device = torch.device("cpu")
torch.set_num_threads(1)

MODEL_URL = "https://huggingface.co/PaniChendra/retinal-oct-model/resolve/main/Backend/MyProject_effcbam_model.pth"
RF_URL = "https://huggingface.co/PaniChendra/retinal-oct-model/resolve/main/Backend/MyProject_rf_model.pkl"
CLASSES_URL = "https://huggingface.co/PaniChendra/retinal-oct-model/resolve/main/Backend/MyProject_classes.json"

model = None
rf = None
class_names = None
original_fc = None
loaded = False


# ================= CBAM =================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max], dim=1)))


class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# ================= MODEL =================
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


# ================= DOWNLOAD =================
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)


# ================= LOAD =================
def load_models():
    global model, rf, class_names, original_fc, loaded

    if loaded:
        return

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

    loaded = True
    print("Models loaded")


# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485]*3, [0.229]*3)
])


# ================= PREDICT =================
def predict_only(image_path):
    load_models()

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.fc = nn.Identity()

    with torch.no_grad():
        features = model(img_tensor).cpu().numpy()

    probs = rf.predict_proba(features)[0]
    pred_idx = np.argmax(probs)

    top2_idx = probs.argsort()[-2:][::-1]
    top2 = [(class_names[i], float(probs[i])) for i in top2_idx]

    model.fc = original_fc

    return class_names[pred_idx], float(probs[pred_idx]), pred_idx, img_tensor, img, top2


# ================= GRADCAM =================
def generate_gradcam(img_tensor, target_class):
    gradients = []
    activations = []

    layer = model.features[-1]

    def fwd(m, i, o):
        activations.append(o)

    def bwd(m, gi, go):
        gradients.append(go[0])

    h1 = layer.register_forward_hook(fwd)
    h2 = layer.register_backward_hook(bwd)

    output = model(img_tensor)
    loss = output[0, target_class]

    model.zero_grad()
    loss.backward()

    grads = gradients[0][0].detach().cpu().numpy()
    acts = activations[0][0].detach().cpu().numpy()

    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)

    #  improved normalization
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    cam = cv2.resize(cam, (224, 224))

    h1.remove()
    h2.remove()

    return cam


# ================= OVERLAY =================
def overlay(img, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img = np.array(img.resize((224, 224)))

    #  better visibility
    return cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)


# ================= PIPELINE =================
def full_pipeline(path):
    pred, conf, idx, tensor, img, top2 = predict_only(path)

    cam = generate_gradcam(tensor, idx)
    result = overlay(img, cam)

    #  RETURN IMAGE (not file path)
    return pred, conf, top2, result