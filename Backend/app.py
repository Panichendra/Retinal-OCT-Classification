from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import base64
import uuid

from PIL import Image
from inference import predict

app = FastAPI()


# =========================
# HOME ROUTE
# =========================
@app.get("/")
def home():
    return {"message": "OCT Model API with GradCAM is running"}


# =========================
# PREDICT ROUTE
# =========================
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    # -------------------------
    # 1. FILE VALIDATION
    # -------------------------
    ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]
    filename = file.filename.lower()

    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid file type. Upload JPG/PNG only."}
        )

    # Unique filename (important for multiple users)
    unique_name = str(uuid.uuid4()) + "_" + file.filename
    file_path = f"temp_{unique_name}"

    # -------------------------
    # 2. SAVE FILE
    # -------------------------
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to save file"}
        )

    # -------------------------
    # 3. IMAGE VALIDATION
    # -------------------------
    try:
        img = Image.open(file_path)
        img.verify()
    except:
        if os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid or corrupted image"}
        )

    # -------------------------
    # 4. MODEL PREDICTION
    # -------------------------
    try:
        prediction, confidence, top2, gradcam_path = predict(file_path)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse(
            status_code=500,
            content={"error": f"Model inference failed: {str(e)}"}
        )

    # -------------------------
    # 5. CONFIDENCE MESSAGE
    # -------------------------
    if confidence < 0.7:
        warning = "Low confidence prediction. Please verify manually."
    else:
        warning = "High confidence prediction"

    # -------------------------
    # 6. CONVERT GRADCAM → BASE64
    # -------------------------
    try:
        with open(gradcam_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
    except:
        encoded = None

    # -------------------------
    # 7. CLEANUP FILES
    # -------------------------
    if os.path.exists(file_path):
        os.remove(file_path)

    if os.path.exists(gradcam_path):
        os.remove(gradcam_path)

    # -------------------------
    # 8. RESPONSE
    # -------------------------
    return JSONResponse({
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "top2_predictions": top2,
        "warning": warning,
        "gradcam_image": encoded   # frontend can display this
    })