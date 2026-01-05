#The prediction of the output is very wrong 

import tensorflow as tf
import numpy as np
import cv2
import os

# --- PATHS ---
IMAGE = "Dataset/test/sad/im0.png"
MODEL_PATH = "/Users/aysoni/Documents/Emotion-Detection/main/emotion_resnet_best.keras"

def build_emotion_model():
    """Manually reconstructs the model architecture used in training."""
    # Use the same base model as your training script
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=(224, 224, 3), include_top=False, weights=None
    )
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    return model

def classify_image_by_path(img_path, model_path):
    EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # 1. Verify files
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    # 2. Reconstruct and Load Weights
    print("Reconstructing architecture and loading weights...")
    try:
        model = build_emotion_model()
        # Loading weights from the .keras file directly
        model.load_weights(model_path)
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    # 3. Preprocess
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Error: Could not read image.")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    img_normalized = img_resized.astype('float32') / 255.0
    input_tensor = np.expand_dims(img_normalized, axis=0)

    # 4. Predict
    print("Analyzing...")
    preds = model(input_tensor, training=False)
    preds_array = preds.numpy()[0]
    
    result_idx = np.argmax(preds_array)
    label = EMOTION_LABELS[result_idx]
    
    print("\n" + "="*35)
    print(f"RESULT: {label} ({preds_array[result_idx]*100:.2f}%)")
    print("="*35)
    return label

if __name__ == "__main__":
    classify_image_by_path(IMAGE, MODEL_PATH)