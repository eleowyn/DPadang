# feedback.py
import os
import shutil
import random
from train_cnn import train_model

def run_retrain():
    """Menjalankan ulang training dengan data feedback menggunakan fine-tuning"""
    print("🚀 Mulai retrain dengan feedback data...")
    try:
        # Sync feedback data ke validation folder (copy beberapa untuk evaluation)
        sync_feedback_to_validation()
        
        # Fine-tune model dengan feedback data (learning rate lebih tinggi)
        train_model(
            data_dir='data/processed',
            model_save_path='models/saved_models/padang_food_model.keras',
            is_finetuning=True  # Enable fine-tuning mode
        )
        print("✅ Retrain selesai.")
    except Exception as e:
        print(f"❌ Error saat retrain: {e}")
        import traceback
        traceback.print_exc()


def sync_feedback_to_validation():
    """Sync feedback images dari train ke validation folder untuk evaluation"""
    print("📊 Syncing feedback data ke validation folder...")
    
    train_dir = 'data/processed/train'
    val_dir = 'data/processed/validation'
    
    if not os.path.exists(train_dir):
        print(f"⚠️ Train dir tidak ada: {train_dir}")
        return
    
    # Untuk setiap kelas di train, copy beberapa feedback images ke validation
    for class_name in os.listdir(train_dir):
        class_train_path = os.path.join(train_dir, class_name)
        class_val_path = os.path.join(val_dir, class_name)
        
        if not os.path.isdir(class_train_path):
            continue
        
        # Pastikan folder validation kelas ada
        os.makedirs(class_val_path, exist_ok=True)
        
        # Cari semua feedback images (dimulai dengan "feedback_")
        feedback_images = [f for f in os.listdir(class_train_path) 
                          if f.startswith('feedback_')]
        
        if feedback_images:
            print(f"  📁 {class_name}: Found {len(feedback_images)} feedback images")
            
            # Copy feedback images ke validation folder
            # Copy semua feedback images (bukan sampling) untuk better evaluation
            for feedback_img in feedback_images:
                src = os.path.join(class_train_path, feedback_img)
                dst = os.path.join(class_val_path, feedback_img)
                
                # Hanya copy jika belum ada
                if not os.path.exists(dst):
                    try:
                        shutil.copy2(src, dst)
                        print(f"    ✓ Copied: {feedback_img}")
                    except Exception as e:
                        print(f"    ✗ Error copying {feedback_img}: {e}")
    
    print("✅ Feedback data synced to validation folder")