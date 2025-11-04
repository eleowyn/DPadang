# validation.py
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file
load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")  # atau gunakan "gemini-2.5-flash"
    print("✅ Model Gemini berhasil diinisialisasi")
else:
    model = None
    print("⚠️ GEMINI_API_KEY tidak ditemukan — validasi akan dilewati.")

def is_food_image(image_path):
    """Gunakan Gemini untuk memastikan gambar adalah makanan."""
    if not model:
        return True  # skip validasi jika tidak ada key

    try:
        uploaded = genai.upload_file(image_path)
        prompt = "Apakah gambar ini berisi makanan? Jawab hanya 'ya' atau 'tidak'."
        result = model.generate_content([prompt, uploaded])
        answer = result.text.strip().lower()
        is_food = bool(re.search(r"\b(ya|makanan|food|dish|meal)\b", answer))
        return is_food
    except Exception as e:
        print(f"❌ Error validasi: {e}")
        return True