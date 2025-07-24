import os
import shutil
import base64
import numpy as np
from io import BytesIO

import cv2
import easyocr
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# --- App setup ---
app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# --- Load ML Models ---
yolo_model = YOLO("C:/works/ocr_updated/best.pt")  # Change to your model path
ocr_reader = easyocr.Reader(['en'], gpu=False)

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Displays the upload page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    """Handles image upload, YOLO detection, OCR reading, and displays results."""
    try:
        # Validate file format
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load image
        image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # YOLO object detection
        results = yolo_model(image_rgb)
        ocr_results_combined = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes: x1, y1, x2, y2

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cropped_img = image[y1:y2, x1:x2]

                # --- Preprocessing for OCR ---
                preprocessed_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

                # Resize small images
                h, w = preprocessed_img.shape[:2]
                if h < 30 or w < 30:
                    preprocessed_img = cv2.resize(preprocessed_img, (max(100, w * 2), max(30, h * 2)))

                # Optional denoising
                preprocessed_img = cv2.fastNlMeansDenoisingColored(preprocessed_img, None, 10, 10, 7, 21)

                # OCR
                ocr_output = ocr_reader.readtext(preprocessed_img)

                # Debug print
                print("OCR Output:", ocr_output)

                for _, text, conf in ocr_output:
                    ocr_results_combined.append(f"{text} (Confidence: {conf:.2f})")

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode()

        return templates.TemplateResponse("results.html", {
            "request": request,
            "results": ocr_results_combined,
            "image_data": img_base64
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
