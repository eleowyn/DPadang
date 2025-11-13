import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from gradcam import GradCAM, visualize_predictions_with_gradcam
import traceback

MODEL_PATH = "models/saved_models/padang_food_model.keras"
CLASS_INDICES_PATH = "models/class_indices.json"

_model = None
_class_map = {}
_gradcam = None

def load_model():
    """Load model dan class indices"""
    global _model, _class_map, _gradcam
    
    try:
        print(f"📦 Loading model dari: {MODEL_PATH}")
        
        # Cek apakah file model ada
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
        
        # Load model
        _model = tf.keras.models.load_model(MODEL_PATH)
        
        # Build the model by running a dummy prediction
        # This ensures model.input and model.output are defined
        dummy_input = tf.zeros((1, 224, 224, 3))
        _ = _model(dummy_input, training=False)
        
        if isinstance(_model, tf.keras.Sequential):
            print("✅ Model berhasil dimuat (Sequential)")
        else:
            print("✅ Model berhasil dimuat (Functional)")
        
        # Try to access input shape (may fail for some model types)
        try:
            print(f"✅ Model built with input shape: {_model.input.shape}")
        except:
            print(f"✅ Model built successfully")
        
        # Load class indices
        print(f"📦 Loading class indices dari: {CLASS_INDICES_PATH}")
        if not os.path.exists(CLASS_INDICES_PATH):
            raise FileNotFoundError(f"Class indices tidak ditemukan: {CLASS_INDICES_PATH}")
        
        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            class_indices = json.load(f)
        
        _class_map = {v: k for k, v in class_indices.items()}
        print(f"✅ Loaded {len(_class_map)} kelas makanan")
        
        # Initialize GradCAM
        _gradcam = GradCAM(_model)
        print(f"✅ GradCAM initialized (using layer: {_gradcam.layer_name})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        raise


def predict_food(image_path, include_gradcam=False):
    """
    Prediksi makanan dari gambar
    
    Args:
        image_path: Path ke gambar
        include_gradcam: Boolean, apakah generate GradCAM
    
    Returns:
        List of predictions dengan format:
        [{"class": "nama_makanan", "confidence": 0.95, "gradcam_path": "..."}, ...]
    """
    global _model, _class_map, _gradcam
    
    try:
        # Load model jika belum dimuat
        if _model is None:
            print("⚠️ Model belum dimuat, loading...")
            load_model()
        
        print(f"🔍 Predicting image: {image_path}")
        
        # Validasi file gambar
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")
        
        # Preprocess image
        print("📸 Preprocessing image...")
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)
        
        # Prediksi
        print("🤖 Running prediction...")
        preds = _model.predict(arr, verbose=0)[0]
        
        # Get top 3 predictions
        top_indices = preds.argsort()[-3:][::-1]
        results = []
        
        for idx in top_indices:
            class_name = _class_map[idx].replace('_', ' ').title()
            confidence = float(preds[idx])
            
            results.append({
                "class": class_name,
                "confidence": confidence
            })
        
        print(f"✅ Prediction complete: {results[0]['class']} ({results[0]['confidence']*100:.1f}%)")
        
        # Tambahkan GradCAM jika diminta
        if include_gradcam:
            print("🔥 Generating GradCAM visualizations...")
            try:
                # Check if model can support GradCAM by testing input access
                try:
                    _ = _model.input
                    print("✅ Model supports GradCAM")
                except AttributeError:
                    print("⚠️ Model tidak support GradCAM, skipping...")
                    return results
                
                gradcam_results = visualize_predictions_with_gradcam(
                    model=_model,
                    img_path=image_path,
                    class_map=_class_map,
                    top_k=3,
                    save_dir="static/gradcam"
                )
                
                # Merge GradCAM paths ke results
                for i, result in enumerate(results):
                    if i < len(gradcam_results):
                        result['gradcam_path'] = gradcam_results[i]['gradcam_path']
                
                print("✅ GradCAM generated successfully")
                
            except Exception as e:
                print(f"⚠️ Error generating GradCAM: {e}")
                traceback.print_exc()
                # Continue without GradCAM
        
        return results
        
    except Exception as e:
        print(f"❌ Error in predict_food: {e}")
        traceback.print_exc()
        raise


def reload_model():
    """Reload model (digunakan setelah retrain)"""
    global _model, _class_map, _gradcam
    
    print("🔄 Reloading model...")
    _model = None
    _class_map = {}
    _gradcam = None
    
    load_model()
    print("✅ Model reloaded successfully")
    
    return True