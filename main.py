# main.py
import os
import shutil
import threading
import time
from flask import Flask, render_template, request, jsonify
from predictor import predict_food, reload_model
from validation import is_food_image
from feedback import run_retrain

app = Flask(__name__)

# Path sesuai struktur folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    finally:
        _retrain_running = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Prediksi makanan"""
    if "image" not in request.files:
        return jsonify({"success": False, "error": "Tidak ada file upload."})

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Nama file kosong."})

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    # Validasi apakah gambar adalah makanan
    if not is_food_image(save_path):
        return jsonify({
            "success": True,
            "is_food": False,
            "predictions": [{"class": "Bukan Makanan", "confidence": 1.0}],
            "image_path": save_path
        })

    preds = predict_food(save_path)
    return jsonify({
        "success": True,
        "is_food": True,
        "predictions": preds,
        "image_path": save_path
    })


@app.route("/feedback", methods=["POST"])
def feedback_route():
    """Menerima feedback dari user & trigger retrain"""
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
        if image_path and os.path.exists(image_path):
            # Generate unique filename
            timestamp = int(time.time())
            filename = f"feedback_{timestamp}_{os.path.basename(image_path)}"
            dest_path = os.path.join(target_dir, filename)
            shutil.copy(image_path, dest_path)
            print(f"Gambar disimpan ke: {dest_path}")

            # Jalankan retrain di thread background
            threading.Thread(target=background_retrain, daemon=True).start()

    return jsonify({
        "success": True,
        "message": "Feedback diterima. Model akan dilatih ulang di background."
    })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)