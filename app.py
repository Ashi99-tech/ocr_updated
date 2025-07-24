from flask import Flask, request, jsonify
import os
import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    image_file = request.files['image']
    model_file = request.files['model']

    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    model_path = os.path.join(UPLOAD_FOLDER, model_file.filename)

    image_file.save(image_path)
    model_file.save(model_path)

    # Load model and image
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)

    reader = easyocr.Reader(['en'], gpu=False)
    output = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped = image[y1:y2, x1:x2]
            ocr_results = reader.readtext(cropped)
            for _, text, conf in ocr_results:
                output.append({"box": [x1, y1, x2, y2], "text": text, "confidence": round(conf, 2)})

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
