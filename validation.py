# validation.py
import os
import re
import cv2
import numpy as np
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
    print("⚠️ GEMINI_API_KEY tidak ditemukan — validasi akan menggunakan fallback method.")


def is_valid_food_image_basic(image_path):
    """
    Fallback validation: Cek apakah gambar memiliki karakteristik makanan
    - Check color histogram (food biasanya punya warm colors)
    - Check image quality
    - Check if it's a real photo (not text/document)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # Convert to HSV untuk analisis warna
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Food biasanya punya warm colors (orange, brown, red, yellow)
        # Hue range untuk warm colors: 0-30 (red) dan 10-40 (orange/yellow) dan 120-180 (green untuk salad)
        lower_warm = np.array([0, 30, 30])
        upper_warm = np.array([40, 255, 255])
        mask_warm = cv2.inRange(hsv, lower_warm, upper_warm)
        
        lower_green = np.array([40, 30, 30])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Hitung persentase warm/food-like colors
        total_pixels = img.shape[0] * img.shape[1]
        warm_pixels = np.sum(mask_warm > 0)
        green_pixels = np.sum(mask_green > 0)
        food_color_ratio = (warm_pixels + green_pixels) / total_pixels
        
        # Food biasanya punya 30% atau lebih warm/food colors
        has_food_colors = food_color_ratio > 0.25
        
        # Check edge density (makanan biasanya punya texture/edges)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels
        has_texture = edge_density > 0.05
        
        # Combination check
        return has_food_colors and has_texture
        
    except Exception as e:
        print(f"⚠️ Error in basic validation: {e}")
        return True  # default ke True jika error


def is_food_image(image_path):
    """
    Gunakan Gemini untuk memastikan gambar adalah makanan.
    Jika Gemini gagal (quota), gunakan fallback method.
    """
    if not model:
        print("⚠️ Gemini tidak tersedia, menggunakan fallback validation...")
        return is_valid_food_image_basic(image_path)

    try:
        uploaded = genai.upload_file(image_path)
        prompt = "Apakah gambar ini berisi makanan yang dapat dimakan? Jawab hanya 'ya' atau 'tidak'."
        result = model.generate_content([prompt, uploaded])
        answer = result.text.strip().lower()
        is_food = bool(re.search(r"\b(ya|makanan|food|dish|meal)\b", answer))
        print(f"✅ Gemini validation: {answer} → {'Makanan' if is_food else 'Bukan Makanan'}")
        return is_food
    except Exception as e:
        print(f"❌ Error validasi Gemini: {e}")
        print("⚠️ Fallback ke basic validation...")
        # Fallback ke basic validation
        return is_valid_food_image_basic(image_path)