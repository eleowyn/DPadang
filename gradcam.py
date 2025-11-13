# gradcam.py - ULTRA FIXED VERSION 🔥
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os


def get_img_array(img_path, size=(224, 224)):
    """Load and preprocess image"""
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array / 255.0


def find_last_conv_layer(model):
    """Find last Conv2D layer in model (including nested models like MobileNetV2)"""
    print("🔍 Searching for last Conv2D layer...")

    def get_all_layers_recursive(model_or_layer):
        """Recursively get all layers including nested models"""
        result = []
        if hasattr(model_or_layer, 'layers'):
            for layer in model_or_layer.layers:
                result.append(layer)
                # Recursively get layers from nested models
                result.extend(get_all_layers_recursive(layer))
        return result

    all_layers = get_all_layers_recursive(model)
    
    # Print model structure for debugging
    print(f"📊 Total layers found: {len(all_layers)}")
    
    # Find the last Conv2D layer
    last_conv_layer = None
    for layer in reversed(all_layers):
        layer_type = layer.__class__.__name__
        # Check if it's a Conv layer - be more lenient in detection
        if 'Conv' in layer_type:
            try:
                # Get output shape - it's already a tuple in Keras 3
                if hasattr(layer, 'output'):
                    output_shape = layer.output.shape
                    # Convert to list if it's not already
                    if hasattr(output_shape, 'as_list'):
                        output_shape = output_shape.as_list()
                    else:
                        output_shape = list(output_shape) if not isinstance(output_shape, (list, tuple)) else output_shape
                    
                    if len(output_shape) == 4:
                        last_conv_layer = layer
                        print(f"✅ Found last conv layer: {layer.name} ({layer_type})")
                        print(f"   Output shape: {output_shape}")
                        break
            except Exception as e:
                continue

    if last_conv_layer is None:
        raise ValueError("❌ No Conv2D layer found in model!")
    
    return last_conv_layer


def make_gradcam_heatmap(img_array, model, last_conv_layer, pred_index=None):
    """
    Generate GradCAM heatmap using GradientTape
    Compatible with Keras 3 and nested functional models (like MobileNetV2)
    Uses a custom gradient computation approach that works with nested models
    
    Args:
        img_array: Preprocessed image array
        model: Keras model
        last_conv_layer: The actual layer object (not name)
        pred_index: Index of class to visualize (None = predicted class)
    
    Returns:
        numpy array: Heatmap
    """
    print("    🔧 Computing GradCAM gradients...")

    # Find the base model (MobileNetV2) that contains the conv layer
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # It's a nested model
            all_nested_layers = []
            def collect_layers(m):
                if hasattr(m, 'layers'):
                    for l in m.layers:
                        all_nested_layers.append(l)
                        collect_layers(l)
            collect_layers(layer)
            
            if last_conv_layer in all_nested_layers:
                base_model = layer
                print(f"    ✅ Found base model: {base_model.name}")
                break
    
    if base_model is None:
        print("    ⚠️ Conv layer not in nested model, using standard approach")
        # Standard GradCAM approach
        grad_model = keras.models.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
    else:
        # For nested models in Keras 3, we use a different approach
        # We'll compute gradients by watching the base model output
        print(f"    🔄 Using nested model GradCAM approach...")
        
        with tf.GradientTape(persistent=True) as tape:
            # Convert input to tensor and watch it
            img_tensor = tf.convert_to_tensor(img_array)
            tape.watch(img_tensor)
            
            # Forward pass through base model
            base_output = base_model(img_tensor, training=False)
            tape.watch(base_output)
            
            # Forward pass through remaining layers
            x = base_output
            for i, layer in enumerate(model.layers):
                if layer == base_model:
                    continue
                # Skip InputLayer
                if isinstance(layer, keras.layers.InputLayer):
                    continue
                # Try to pass training=False if layer supports it, otherwise just call it
                try:
                    x = layer(x, training=False)
                except TypeError:
                    x = layer(x)
            
            predictions = x
            
            # Get predicted class
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        # Compute gradients w.r.t. base model output
        grads = tape.gradient(class_channel, base_output)
        conv_outputs = base_output
        
        del tape  # Clean up persistent tape
    
    if grads is None:
        print("    ⚠️ Gradients are None, GradCAM cannot be computed")
        return None
    
    # Ensure we have 4D tensors (batch, height, width, channels)
    if len(conv_outputs.shape) != 4:
        print(f"    ⚠️ Conv output shape is {conv_outputs.shape}, expected 4D")
        # If it's 2D (from GlobalAveragePooling), we can't create a spatial heatmap
        return None
    
    # Pool the gradients across the spatial dimensions  
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Get the first (and only) sample
    conv_outputs = conv_outputs[0]
    
    # Multiply each channel by "how important it is"
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    
    print(f"    ✅ Heatmap generated, shape: {heatmap.shape}")
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, save_path, alpha=0.5):
    """Overlay heatmap on image and save"""
    if heatmap is None or heatmap.size == 0:
        print("    ⚠️ Heatmap is None or empty, skipping save")
        return None
    
    # Load original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose the heatmap on original image
    superimposed_img = colored_heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Save the result
    result = Image.fromarray(superimposed_img)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result.save(save_path)

    print(f"    ✅ GradCAM saved to: {save_path}")
    return result


def visualize_predictions_with_gradcam(model, img_path, class_map, top_k=3, save_dir="static/gradcam"):
    """
    Generate GradCAM visualizations for top-k predictions
    
    Args:
        model: Trained Keras model
        img_path: Path to image
        class_map: Dictionary mapping class indices to class names
        top_k: Number of top predictions to visualize
        save_dir: Directory to save GradCAM images
    
    Returns:
        List of dicts with rank, class, confidence, gradcam_path
    """
    print(f"\n🔥 Generating GradCAM visualizations for: {os.path.basename(img_path)}")

    os.makedirs(save_dir, exist_ok=True)

    # Step 1: Find the last convolutional layer
    try:
        last_conv_layer = find_last_conv_layer(model)
        print(f"🎯 Using conv layer: {last_conv_layer.name}")
    except Exception as e:
        print(f"❌ Error finding conv layer: {e}")
        import traceback
        traceback.print_exc()
        return []

    # Step 2: Preprocess image and get predictions
    img_array = get_img_array(img_path)
    print("🤖 Running predictions...")
    preds = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(preds)[-top_k:][::-1]

    results = []
    base_filename = os.path.splitext(os.path.basename(img_path))[0]

    # Step 3: Generate GradCAM for each top prediction
    for rank, idx in enumerate(top_indices, 1):
        try:
            class_name = class_map[idx].replace('_', ' ').replace('-', ' ')
            confidence = preds[idx]
            print(f"  #{rank} {class_name} ({confidence*100:.1f}%)")

            # Generate heatmap
            heatmap = make_gradcam_heatmap(
                img_array=img_array,
                model=model,
                last_conv_layer=last_conv_layer,  # Pass the actual layer object
                pred_index=idx
            )

            if heatmap is None:
                print(f"    ⚠️ Heatmap generation failed, skipping...")
                continue

            # Save GradCAM visualization
            safe_name = class_name.replace(' ', '_').replace('/', '_')
            save_path = os.path.join(
                save_dir,
                f"{base_filename}_rank{rank}_{safe_name}.png"
            )

            result = save_and_display_gradcam(img_path, heatmap, save_path, alpha=0.5)
            
            if result is not None:
                results.append({
                    'rank': rank,
                    'class': class_name,
                    'confidence': float(confidence),
                    'gradcam_path': save_path
                })
            else:
                print(f"    ⚠️ Failed to save GradCAM")

        except Exception as e:
            print(f"    ⚠️ Error processing rank {rank}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"✅ Successfully generated {len(results)}/{top_k} visualizations\n")
    return results


class GradCAM:
    """Backward-compatible GradCAM class for direct usage"""
    
    def __init__(self, model, layer_name=None):
        """
        Initialize GradCAM
        
        Args:
            model: Keras model
            layer_name: Name of conv layer (optional, will auto-detect)
        """
        self.model = model
        
        if layer_name:
            # Find the layer by name
            def find_layer_by_name(layers, name):
                for layer in layers:
                    if layer.name == name:
                        return layer
                    if hasattr(layer, 'layers'):
                        found = find_layer_by_name(layer.layers, name)
                        if found:
                            return found
                return None
            
            self.layer = find_layer_by_name(model.layers, layer_name)
            if self.layer is None:
                raise ValueError(f"Layer '{layer_name}' not found in model")
            self.layer_name = layer_name
        else:
            # Auto-detect last conv layer
            self.layer = find_last_conv_layer(model)
            self.layer_name = self.layer.name
        
        print(f"🎯 GradCAM initialized with layer: {self.layer_name}")

    def save_gradcam(self, img_path, save_path, pred_index=None, alpha=0.5):
        """
        Generate and save GradCAM visualization
        
        Args:
            img_path: Path to input image
            save_path: Path to save GradCAM output
            pred_index: Class index to visualize (None = predicted class)
            alpha: Overlay transparency (0-1)
        
        Returns:
            PIL Image object or None
        """
        img_array = get_img_array(img_path)
        heatmap = make_gradcam_heatmap(
            img_array=img_array,
            model=self.model,
            last_conv_layer=self.layer,
            pred_index=pred_index
        )
        return save_and_display_gradcam(img_path, heatmap, save_path, alpha)

