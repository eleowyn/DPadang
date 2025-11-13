#!/usr/bin/env python3
"""
Test script to verify GradCAM works with the MobileNetV2 model
"""

import os
import sys
import tensorflow as tf
import json
from gradcam import visualize_predictions_with_gradcam, find_last_conv_layer

def test_gradcam():
    print("="*70)
    print("🧪 TESTING GRADCAM FUNCTIONALITY")
    print("="*70)
    
    # Load model
    model_path = "models/saved_models/padang_food_model.keras"
    class_indices_path = "models/class_indices.json"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    if not os.path.exists(class_indices_path):
        print(f"❌ Class indices not found: {class_indices_path}")
        return False
    
    print(f"\n📦 Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully")
    
    # Load class indices
    print(f"\n📦 Loading class indices from: {class_indices_path}")
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    class_map = {v: k for k, v in class_indices.items()}
    print(f"✅ Loaded {len(class_map)} classes")
    
    # Print model structure
    print("\n📊 Model Structure:")
    print(f"Model Type: {type(model).__name__}")
    print(f"Number of layers: {len(model.layers)}")
    print("\nLayers:")
    for i, layer in enumerate(model.layers):
        print(f"  {i}: {layer.name} - {layer.__class__.__name__}")
        if hasattr(layer, 'layers') and len(layer.layers) > 0:
            print(f"      ↳ Nested model with {len(layer.layers)} layers")
    
    # Test finding conv layer
    print("\n🔍 Testing conv layer detection...")
    try:
        last_conv = find_last_conv_layer(model)
        print(f"✅ Successfully found: {last_conv.name}")
    except Exception as e:
        print(f"❌ Failed to find conv layer: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Find a test image
    print("\n🔍 Looking for test images...")
    test_dirs = [
        "data/processed/test",
        "data/processed/validation",
        "static/uploads"
    ]
    
    test_image = None
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_image = os.path.join(root, file)
                        break
                if test_image:
                    break
        if test_image:
            break
    
    if not test_image:
        print("❌ No test image found!")
        print("Please add a test image to one of these directories:")
        for d in test_dirs:
            print(f"  - {d}")
        return False
    
    print(f"✅ Found test image: {test_image}")
    
    # Test GradCAM generation
    print("\n🔥 Testing GradCAM generation...")
    try:
        results = visualize_predictions_with_gradcam(
            model=model,
            img_path=test_image,
            class_map=class_map,
            top_k=3,
            save_dir="static/gradcam"
        )
        
        if len(results) > 0:
            print(f"\n✅ SUCCESS! Generated {len(results)} GradCAM visualizations")
            print("\nResults:")
            for result in results:
                print(f"  Rank {result['rank']}: {result['class']} "
                      f"({result['confidence']*100:.1f}%)")
                print(f"    → {result['gradcam_path']}")
            return True
        else:
            print("❌ No GradCAM visualizations were generated")
            return False
            
    except Exception as e:
        print(f"❌ Error generating GradCAM: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n")
    success = test_gradcam()
    print("\n" + "="*70)
    if success:
        print("✅ ALL TESTS PASSED! GradCAM is working correctly! 🎉")
    else:
        print("❌ TESTS FAILED! Please check the errors above.")
    print("="*70 + "\n")
    
    sys.exit(0 if success else 1)
