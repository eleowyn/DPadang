# train_efficientb0.py
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001

def create_model(num_classes):
    print("🔨 Membuat model...")
    base_model = EfficientNetB0(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    print("✅ Model berhasil dibuat!")
    return model

def train_model(train_dir, val_dir, model_save_path):
    print("\n" + "="*50)
    print("🚀 MULAI TRAINING MODEL")
    print("="*50 + "\n")

    # Cek folder training & validation
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"❌ Folder train/validation tidak ditemukan!")
        print(f"Pastikan struktur seperti ini:")
        print("padangfood/")
        print(" ├── train/")
        print(" └── validation/")
        return

    # Data generator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    print("📂 Loading dataset...")

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    num_classes = len(train_generator.class_indices)
    print(f"\n✅ Dataset loaded!")
    print(f"   📊 Jumlah kelas: {num_classes}")
    print(f"   📸 Training images: {train_generator.samples}")
    print(f"   📸 Validation images: {val_generator.samples}\n")

    model = create_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1),
        keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Simpan class indices
    class_indices_path = 'models/class_indices.json'
    os.makedirs("models", exist_ok=True)
    with open(class_indices_path, 'w') as f:
        json.dump(train_generator.class_indices, f, indent=4)
    print(f"📝 Class indices disimpan di: {class_indices_path}")

    plot_training_history(history)
    print(f"✅ Training selesai! Model disimpan di: {model_save_path}")

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig('models/training_history.png')
    plt.close()
    print("📊 Grafik training disimpan di: models/training_history.png")

if __name__ == '__main__':
    # Path disesuaikan dengan struktur kamu
    train_path = 'padangfood/train'
    val_path = 'padangfood/validation'
    model_path = 'models/saved_models/padang_food_modelefficientb0.keras'

    train_model(train_path, val_path, model_path)