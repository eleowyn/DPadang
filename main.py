import os
import shutil
import threading
import time
import traceback
from flask import Flask, render_template, request, jsonify, send_from_directory
from predictor import predict_food, reload_model
from validation import is_food_image
from feedback import run_retrain

app = Flask(__name__)

# Path sesuai struktur folder
UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

FEEDBACK_LOG = "feedback_log.txt"
DATASET_PATH = "data/processed/train"

# Flag supaya retrain tidak dijalankan bersamaan
_retrain_running = False


def background_retrain():
    """Menjalankan retraining model di thread terpisah"""
    global _retrain_running
    if _retrain_running:
        print("Retrain sedang berjalan, dilewati sementara.")
        return

    _retrain_running = True
    try:
        print("Memulai retrain model...")
        run_retrain()
        print("Retrain selesai. Reload model...")
        reload_model()
    except Exception as e:
        print("Error retrain:", e)
        traceback.print_exc()
    finally:
        _retrain_running = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Prediksi makanan dengan optional GradCAM"""
    try:
        # Validasi file upload
        if "image" not in request.files:
            return jsonify({"success": False, "error": "Tidak ada file upload."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Nama file kosong."}), 400

        # Validasi ukuran file
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset pointer
        
        if file_size > 16 * 1024 * 1024:
            return jsonify({"success": False, "error": "Ukuran file terlalu besar! Maksimal 16MB"}), 400

        # Always generate GradCAM automatically
        include_gradcam = True

        # Generate unique filename
        timestamp = int(time.time())
        # Sanitize filename
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
        filename = f"{timestamp}_{safe_filename}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Simpan file
        file.save(save_path)
        print(f"✅ File disimpan: {save_path}")

        # Validasi apakah gambar adalah makanan
        print(f"🔍 Validasi gambar dengan Gemini...")
        if not is_food_image(save_path):
            print("❌ Bukan makanan terdeteksi")
            return jsonify({
                "success": True,
                "is_food": False,
                "predictions": [{"class": "Bukan Makanan", "confidence": 1.0}],
                "image_path": f"/static/uploads/{filename}"
            })

        print(f"✅ Validasi berhasil, memulai prediksi...")
        
        # Prediksi
        preds = predict_food(save_path, include_gradcam=include_gradcam)
        
        print(f"✅ Prediksi selesai: {preds}")
        
        # Convert absolute paths to relative URLs
        for pred in preds:
            if 'gradcam_path' in pred and pred['gradcam_path']:
                # Normalize path separators dan convert ke relative URL
                gradcam_path = pred['gradcam_path'].replace('\\', '/')
                if not gradcam_path.startswith('/'):
                    gradcam_path = '/' + gradcam_path
                pred['gradcam_path'] = gradcam_path
        
        return jsonify({
            "success": True,
            "is_food": True,
            "predictions": preds,
            "image_path": f"/static/uploads/{filename}"
        })
    
    except Exception as e:
        # Log error detail
        print("="*60)
        print("ERROR PADA /predict:")
        print("="*60)
        print(f"Error: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        print("="*60)
        
        return jsonify({
            "success": False, 
            "error": f"Error server: {str(e)}"
        }), 500


@app.route("/feedback", methods=["POST"])
def feedback_route():
    """Menerima feedback dari user & trigger retrain"""
    try:
        data = request.get_json()
        prediction = data.get("prediction")
        correct = data.get("correct")
        correct_label = data.get("correct_label")
        image_path = data.get("image_path")

        # Catat feedback
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.asctime()} | pred={prediction} | correct={correct} | label={correct_label}\n")

        # Jika prediksi salah dan user memberikan label benar
        if not correct and correct_label:
            # Normalisasi label: "Bukan Makanan" -> "bukan_makanan"
            label_normalized = correct_label.lower().replace(" ", "_")
            target_dir = os.path.join(DATASET_PATH, label_normalized)
            os.makedirs(target_dir, exist_ok=True)

            # Simpan gambar untuk retrain
            # Convert URL path to file path
            if image_path and image_path.startswith('/static/'):
                image_path = image_path[1:]  # Remove leading /
            
            if image_path and os.path.exists(image_path):
                # Generate unique filename
                timestamp = int(time.time())
                filename = f"feedback_{timestamp}_{os.path.basename(image_path)}"
                dest_path = os.path.join(target_dir, filename)
                shutil.copy(image_path, dest_path)
                print(f"✅ Gambar disimpan ke: {dest_path}")

                # Jalankan retrain di thread background
                threading.Thread(target=background_retrain, daemon=True).start()
                
                return jsonify({
                    "success": True,
                    "message": f"Feedback diterima dan disimpan ke '{label_normalized}'. Model akan dilatih ulang di background."
                })

        return jsonify({
            "success": True,
            "message": "Terima kasih atas feedback Anda!"
        })
    
    except Exception as e:
        print(f"Error pada /feedback: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/ask-llm", methods=["POST"])
def ask_llm():
    """Generate recipe, nutritional info, atau informasi lain tentang makanan menggunakan LLM
    
    Uses multiple backends:
    1. Ollama (local, free, unlimited) - preferred
    2. Hugging Face API (cloud, free tier 15k/month) - fallback
    """
    try:
        from llm_provider import LLMProvider
        
        data = request.get_json()
        food_name = data.get("food_name")
        prompt = data.get("prompt")
        
        if not food_name or not prompt:
            return jsonify({
                "success": False,
                "error": "Food name dan prompt diperlukan"
            }), 400
        
        print(f"🤖 LLM Query - Food: {food_name}, Prompt: {prompt}")
        
        # Use LLM provider (tries Ollama first, then HF)
        result = LLMProvider.ask_about_food(food_name, prompt)
        
        if result["success"]:
            provider = result.get("provider", "unknown")
            print(f"✅ LLM Response generated ({len(result['response'])} chars) via {provider}")
            
            return jsonify({
                "success": True,
                "response": result["response"],
                "food_name": food_name,
                "provider": provider
            })
        else:
            error_msg = result.get("error", "Unknown error")
            print(f"❌ LLM Error: {error_msg}")
            
            return jsonify({
                "success": False,
                "error": "LLM tidak tersedia saat ini. Pastikan Ollama berjalan atau setup Hugging Face API.",
                "details": error_msg
            }), 503
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Unexpected error pada /ask-llm: {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "error": "Error saat memproses pertanyaan. Silakan coba lagi.",
            "details": error_msg[:100]
        }), 500


# Serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


# Handle favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == "__main__":
    print("="*60)
    print("🚀 Starting Flask App (main.py)")
    print("="*60)
    print("Checking dependencies...")
    
    # Pre-load model untuk menghindari error saat predict pertama kali
    try:
        from predictor import load_model
        print("📦 Loading model...")
        load_model()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not pre-load model: {e}")
        print("Model will be loaded on first prediction")
    
    print("="*60)
    print("🌐 Server starting on http://0.0.0.0:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)