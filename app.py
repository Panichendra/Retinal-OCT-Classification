from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os

from inference import predict_only, generate_gradcam_only

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Retinal OCT API Running 🚀"}

# =========================
# PREDICT ONLY (FAST)
# =========================

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, confidence, top2, pred_idx, img_tensor, img = predict_only(file_path)

    # store temp info for gradcam (optional improvement later)
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "top2_predictions": top2,
        "note": "Call /gradcam separately for visualization"
    }

# =========================
# GRADCAM ONLY
# =========================

@app.post("/gradcam")
async def gradcam_api(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, confidence, top2, pred_idx, img_tensor, img = predict_only(file_path)

    gradcam_path = generate_gradcam_only(pred_idx, img_tensor, img)

    return FileResponse(gradcam_path)
