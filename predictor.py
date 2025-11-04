# predictor.py
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "models/saved_models/padang_food_model.keras"
CLASS_INDICES_PATH = "models/class_indices.json"

_model = None
_class_map = {}

def load_model():
    global _model, _class_map
    _model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    _class_map = {v: k for k, v in class_indices.items()}
    print("✅ Model dimuat ulang.")

def predict_food(image_path):
    global _model, _class_map
    if _model is None:
        load_model()

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = _model.predict(arr)[0]

    top_indices = preds.argsort()[-3:][::-1]
    return [{"class": _class_map[i], "confidence": float(preds[i])} for i in top_indices]

def reload_model():
    load_model()
    return True