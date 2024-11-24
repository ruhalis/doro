from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model
import uvicorn

app = FastAPI()

MODEL_PATH = "temp_g_model_with_training_config.tf"
g_model = load_model(MODEL_PATH)


def perform_inference(image_data: bytes):
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((256, 256))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        with tf.device('/GPU:0'):
            generated_image = g_model.predict(image_array, steps=100)

        generated_image = (generated_image[0] + 1) * 127.5
        generated_image = np.clip(generated_image, 0, 255).astype(np.uint8)

        result_image = Image.fromarray(generated_image)
        return result_image, None
    except Exception as e:
        return None, str(e)


@app.post("/infer/")
def infer(file: UploadFile = File(...)):
    try:
        image_data = file.file.read()
        result_image, error = perform_inference(image_data)
        if error:
            return {"error": error}

        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
