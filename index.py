import base64
import io
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

from flask import Flask, request, after_this_request


# load model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

app = Flask(__name__)

@app.route('/inference/object', methods=['POST'])
def create_user():
    # Get the request body
    data = request.get_json()

    # Process the request
    threshold = data['threshold']
    image = data['image']

    # Decode the string
    binary_data = base64.b64decode(image)

    # Convert the bytes into a PIL image
    image = Image.open(io.BytesIO(binary_data))

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    count = 0

    entities = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        entities.append({
            "label": model.config.id2label[label.item()],
            "score": round(score.item(), 3),
            "box": {
                "xmin": box[0],
                "ymin": box[1],
                "xmax": box[2],
                "ymax": box[3]
            }
        })

        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )
        count += 1

    print("Total objects detected: ", count)
    
    # Return the response, in JSON format with all the objects

    return entities

# CORS allow any
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response
    

if __name__ == '__main__':
    app.run(port=5000)
