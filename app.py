from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os

from inference import predict_only, generate_gradcam

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def home():
    return {"message": "API running"}

# =========================
# PREDICT
# =========================

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):

    path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, confidence, pred_idx, img_tensor, img = predict_only(path)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }

# =========================
# GRADCAM
# =========================

@app.post("/gradcam")
async def gradcam_api(file: UploadFile = File(...)):

    path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, confidence, pred_idx, img_tensor, img = predict_only(path)

    cam_path = generate_gradcam(pred_idx, img_tensor, img)

    return FileResponse(cam_path)
