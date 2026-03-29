from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os

from Backend.inference import predict

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Retinal OCT API Running 🚀"}


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):

    # Save file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    prediction, confidence, top2, gradcam_path = predict(file_path)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "top2_predictions": top2,
        "gradcam_image_url": f"/gradcam/{gradcam_path}"
    }


@app.get("/gradcam/{filename}")
def get_gradcam(filename: str):
    return FileResponse(filename)
