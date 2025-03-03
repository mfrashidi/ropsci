import time

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import logging
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import math

# Constants
CLASS_LABELS = ["paper", "rock", "scissors"]
ACCEPTANCE_THRESHOLD = 0.6

# Load the model
model = YOLO('model.pt')
model.to('mps')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode a base64 string to an OpenCV image."""
    try:
        img_data = base64.b64decode(base64_string)
        np_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise

@csrf_exempt
def predict_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            # Get the base64 image string
            base64_image = data.get("image")
            if not base64_image:
                return JsonResponse({"error": "No image provided"}, status=400)

            name = data.get("name")
            if not name:
                return JsonResponse({"error": "No name provided"}, status=400)

            game_round = data.get("round")
            if not game_round:
                return JsonResponse({"error": "No round provided"}, status=400)

            game = data.get("game")
            if not game:
                return JsonResponse({"error": "No game provided"}, status=400)

            if base64_image.startswith("data:image"):
                base64_image = base64_image.split(",")[1]

            # Decode and preprocess the image
            image = decode_base64_image(base64_image)
            timestamp = time.time()
            filename = f"detected/{name}_{game}_{game_round}_{timestamp}.jpg"
            cv2.imwrite(filename, image)

            # Run the prediction
            result = model.predict(
                source=filename,
                conf=0.5,
                save=False
            )
            confidences = {
                i: 0 for i in CLASS_LABELS
            }
            for r in result:
                boxes = r.boxes
                for box in boxes:
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    classname = CLASS_LABELS[int(box.cls[0])]
                    if confidence > confidences[classname]:
                        confidences[classname] = confidence

            if not confidences:
                return JsonResponse({"error": "No predictions made"}, status=200)

            return JsonResponse(confidences)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return JsonResponse({"error": "An error occurred during prediction"}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)
