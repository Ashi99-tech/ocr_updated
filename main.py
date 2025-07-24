from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from PIL import Image
import io
import json
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# ✅ Configure Gemini 2.0 Flash
genai.configure(api_key="................")
model = genai.GenerativeModel("gemini-2.0-flash")

# ✅ Load YOLOv8 model
yolo_model = YOLO("C:/Users/ASUS TUF/Downloads/ocr_cropped/ocr_cropped/best.pt")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(image_file).convert("RGB")

    # Detect text regions using YOLO
    results = yolo_model.predict(image, conf=0.4)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    if len(boxes) == 0:
        return jsonify({"error": "No text area detected"}), 404

    # Crop the first detected region
    x1, y1, x2, y2 = boxes[0]
    cropped = image.crop((x1, y1, x2, y2))

    # Convert to bytes
    img_bytes = io.BytesIO()
    cropped.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    # Gemini prompt
    prompt = """Extract all visible text (letters and numbers only) from this image and return it as a JSON like:
    {
      "detected_text": ["..."]
    }"""

    try:
        response = model.generate_content(
            [prompt, Image.open(io.BytesIO(img_bytes))],
            generation_config={"response_mime_type": "application/json"}
        )

        try:
            result_json = json.loads(response.text)
        except json.JSONDecodeError:
            result_json = {"detected_text": [response.text.strip()]}

        return jsonify(result_json)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
