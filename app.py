from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

# Load your YOLO model
model = YOLO('best.pt')

labels_file = 'labels.txt'  # Path to your labels file

@app.route('/', methods=['POST'])
def detect_objects():
    # Check if the request contains an image
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    # Decode base64-encoded image data
    base64_image = request.json['image']
    image_bytes = base64.b64decode(base64_image)

    # Convert image data to PIL Image
    image = Image.open(BytesIO(image_bytes))

    # Run inference on the image
    results = model(image)

    # Read labels from the text file
    labels = read_labels_from_file(labels_file)

    # Extract class IDs and convert them to labels
    detections = []
    for r in results:
        class_ids = r.boxes.cls.cpu().numpy().tolist()   # Extract class IDs from the 6th column
        class_labels = [labels[int(cls_id)] for cls_id in class_ids]  # Convert class IDs to labels

        for class_label in class_labels:
            detection = {
                'class_label': class_label,
            }
            detections.append(detection)

    return jsonify({'detections': detections})

def read_labels_from_file(file_path):
    with open(file_path, 'r') as file:
        labels = [line.strip() for line in file]
    return labels

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
