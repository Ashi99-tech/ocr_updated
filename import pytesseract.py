import easyocr
import cv2

# Load image
image_path = 'C:/works/ocr_updated/1737097070.jpg'  # Replace with actual path
image = cv2.imread(image_path)

# Initialize reader
reader = easyocr.Reader(['en'])

# Perform OCR
results = reader.readtext(image)

# Sort results top to bottom using the top-left y-coordinate
results_sorted = sorted(results, key=lambda x: x[0][0][1])  # x[0][0][1] = top-left corner y

# Print line-by-line
print("Extracted Text:")
for bbox, text, conf in results_sorted:
    print(text)
