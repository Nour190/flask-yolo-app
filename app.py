import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import base64
import gdown

app = Flask(__name__)

gdrive_url = 'https://drive.google.com/uc?id=1nQ2iQN0YEr3NonL0D26xBYqX54o_DaDi'
model_path = 'best.pt'
model = None

def download_model(url, output):
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
        print(f"Downloaded {output} from Google Drive")

@app.route('/', methods=['POST'])
def detect_objects():
    global model
    if model is None:
        download_model(gdrive_url, model_path)
        model = YOLO(model_path)

    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    base64_image = request.json['image']
    image_bytes = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_bytes))
    results = model(image)
    labels = read_labels_from_file('labels.txt')
    detections = []
    for r in results:
        class_ids = r.boxes.cls.cpu().numpy().tolist()
        class_labels = [labels[int(cls_id)] for cls_id in class_ids]
        for class_label in class_labels:
            detections.append({'class_label': class_label})
    return jsonify({'detections': detections})

def read_labels_from_file(file_path):
    with open(file_path, 'r') as file:
        labels = [line.strip() for line in file]
    return labels

if __name__ == '__main__':
    app.run(debug=True)


