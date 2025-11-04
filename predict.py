# predict.py
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

def load_model_and_classes(model_path='models/saved_models/padang_food_model.keras'):
    """Load trained model dan class indices"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan di: {model_path}")
    
    print(f"Loading model dari: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded!")
    
    class_indices_path = 'models/class_indices.json'
    if not os.path.exists(class_indices_path):
        raise FileNotFoundError(f"Class indices tidak ditemukan di: {class_indices_path}")
    
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    
    # Reverse mapping: index -> class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    print(f"Loaded {len(idx_to_class)} kelas makanan")
    
    return model, idx_to_class

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess gambar untuk prediksi"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path, model, idx_to_class, top_k=3):
    """
    Prediksi gambar dan return top-k hasil
    """
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'class': idx_to_class[idx].replace('_', ' ').title(),
            'confidence': float(predictions[idx])
        })
    
    return results