# feedback.py
from train_cnn import train_model

def run_retrain():
    """Menjalankan ulang training"""
    print("🚀 Mulai retrain...")
    try:
        train_model(
            data_dir='data/processed',
            model_save_path='models/saved_models/padang_food_model.keras'
        )
        print("✅ Retrain selesai.")
    except Exception as e:
        print(f"❌ Error saat retrain: {e}")
        import traceback
        traceback.print_exc()