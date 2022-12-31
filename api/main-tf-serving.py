from fastapi import FastAPI, UploadFile, File
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import requests

app = FastAPI()

endpoint = "http://localhost:8501/v1/models/potatoes_model/predict"

CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']


def read_file_as_image(data) -> np.ndarray:
    return np.array(Image.open(BytesIO(data)))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {
        'instances': img_batch.tolist()
    }
    response = requests.post(endpoint, json=json_data)

    predictions = response.json()['predictions'][0]
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)
    return {
        "class": predicted_class,
        "confidence": float(confidence)*100
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
