# train_cnn.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json
from sklearn.utils.class_weight import compute_class_weight

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001


def create_model(num_classes):
    """
    Buat model menggunakan Transfer Learning dengan MobileNetV2
    """
    print("Membuat model...")
    
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    print("Model berhasil dibuat!")
    return model


def train_model(data_dir, model_save_path):
    """
    Training model dengan data augmentation
    """
    print("\n" + "="*50)
    print("MULAI TRAINING MODEL")
    print("="*50 + "\n")
    
    # Cek apakah folder data ada
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        print(f"Error: Folder {train_dir} tidak ditemukan!")
        print("Jalankan preprocessing terlebih dahulu!")
        return None, None
    
    # Data Augmentation untuk training
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
    
    # Validasi hanya rescale
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    print("Loading dataset...")
    
    # Load data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'validation'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    num_classes = len(train_generator.class_indices)
    print(f"\nDataset loaded!")
    print(f"Jumlah kelas: {num_classes}")
    print(f"Training images: {train_generator.samples}")
    print(f"Validation images: {val_generator.samples}")
    print(f"\nKelas makanan:")
    
    # Hitung jumlah gambar per kelas
    class_counts = []
    for class_name, idx in sorted(train_generator.class_indices.items(), key=lambda x: x[1]):
        class_path = os.path.join(train_dir, class_name)
        count = len([f for f in os.listdir(class_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        class_counts.append(count)
        print(f"   {idx}. {class_name}: {count} images")

    # Cek jika ada kelas kosong
    if any(c == 0 for c in class_counts):
        print("\n⚠️ Warning: Ada kelas yang tidak memiliki gambar! Harap periksa dataset.\n")

    # Hitung class weights (versi aman)
    labels = []
    for class_idx, count in enumerate(class_counts):
        if count > 0:  # hanya kelas yang memiliki gambar
            labels.extend([class_idx] * count)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"\nClass weights (untuk handle imbalance):")
    for idx, weight in class_weight_dict.items():
        class_name = [k for k, v in train_generator.class_indices.items() if v == idx][0]
        print(f"   {class_name}: {weight:.2f}")
    
    # Create model
    model = create_model(num_classes)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=10, 
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, 
            patience=5,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            model_save_path, 
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training dengan class weights
    print("\n" + "="*50)
    print("MULAI TRAINING...")
    print("="*50 + "\n")
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Save class indices
    class_indices_path = 'models/class_indices.json'
    os.makedirs(os.path.dirname(class_indices_path), exist_ok=True)
    with open(class_indices_path, 'w') as f:
        json.dump(train_generator.class_indices, f, indent=4)
    
    print(f"\nTraining selesai!")
    print(f"Model disimpan di: {model_save_path}")
    print(f"Class indices disimpan di: {class_indices_path}")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history


def plot_training_history(history):
    """Plot dan simpan grafik training"""
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/training_history.png')
    print(f"Grafik training disimpan di: models/training_history.png")
    plt.close()


if __name__ == '__main__':
    train_model('data/processed', 'models/saved_models/padang_food_model.keras')
