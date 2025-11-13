# train_resnet50.py
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
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
    base_model = ResNet50(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    print("✅ Model berhasil dibuat!")
    print(f"   🧠 Base Model: ResNet50")
    print(f"   📊 Total Parameters: {model.count_params():,}")
    return model

def train_model(train_dir, val_dir, model_save_path):
    print("\n" + "="*50)
    print("🚀 MULAI TRAINING MODEL - ResNet50")
    print("="*50 + "\n")

    # Cek folder training & validation
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"❌ Folder train/validation tidak ditemukan!")
        print(f"Pastikan struktur seperti ini:")
        print("padangfood/")
        print(" ├── train/")
        print(" └── validation/")
        return

    # Data generator dengan augmentasi lebih kuat
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.15,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    print("📂 Loading dataset...")

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    num_classes = len(train_generator.class_indices)
    print(f"\n✅ Dataset loaded!")
    print(f"   📊 Jumlah kelas: {num_classes}")
    print(f"   📸 Training images: {train_generator.samples}")
    print(f"   📸 Validation images: {val_generator.samples}")
    print(f"   🔄 Steps per epoch: {len(train_generator)}\n")

    model = create_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    print("\n🎯 Mulai training...\n")
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
    print(f"\n📝 Class indices disimpan di: {class_indices_path}")

    # Evaluasi final
    print("\n" + "="*50)
    print("📈 EVALUASI MODEL")
    print("="*50)
    val_loss, val_acc, val_top3 = model.evaluate(val_generator, verbose=0)
    print(f"   ✅ Validation Accuracy: {val_acc*100:.2f}%")
    print(f"   ✅ Top-3 Accuracy: {val_top3*100:.2f}%")
    print(f"   📉 Validation Loss: {val_loss:.4f}")

    plot_training_history(history)
    print(f"\n✅ Training selesai! Model disimpan di: {model_save_path}")

def plot_training_history(history):
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
    plt.title('Model Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Model Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Top-3 Accuracy plot
    plt.subplot(1, 3, 3)
    plt.plot(history.history['top_3_accuracy'], label='Train Top-3', linewidth=2)
    plt.plot(history.history['val_top_3_accuracy'], label='Val Top-3', linewidth=2)
    plt.title('Top-3 Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig('models/training_history.png', dpi=150)
    plt.close()
    print("📊 Grafik training disimpan di: models/training_history.png")

if __name__ == '__main__':
    # Path disesuaikan dengan struktur kamu
    train_path = 'padangfood/train'
    val_path = 'padangfood/validation'
    model_path = 'models/saved_models/padang_food_resnet50.keras'

    train_model(train_path, val_path, model_path)