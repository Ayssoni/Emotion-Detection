import os
import ssl
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- STEP 1: MAC SSL FIX (Required for downloading ResNet weights) ---
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and 
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# --- STEP 2: PATHS (Absolute paths for your Mac) ---
TRAIN_DIR = '/Users/aysoni/Documents/Emotion-Detection/Dataset/train'
TEST_DIR = '/Users/aysoni/Documents/Emotion-Detection/Dataset/test'

# --- STEP 3: DATA GENERATORS (With 224x224 Upscaling) ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training'
)
val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation'
)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
)

# --- STEP 4: MODEL ARCHITECTURE ---
# If a checkpoint exists, load it; otherwise, build a new model
checkpoint_path = "emotion_checkpoint.keras"

if os.path.exists(checkpoint_path):
    print("\n--- Found Checkpoint! Resuming from previous state ---")
    model = tf.keras.models.load_model(checkpoint_path)
else:
    print("\n--- Building New ResNet50V2 Model ---")
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet'
    )
    base_model.trainable = True # Fine-tune the whole model for accuracy

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(7, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

# --- STEP 5: SMART CALLBACKS & CHECKPOINTS ---
# 1. Checkpoint: Saves the whole model every time accuracy improves
ckpt_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# 2. Early Stopping and LR Reduction
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)

# --- STEP 6: TRAINING ---
print("\n--- Starting Training Process ---")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[ckpt_callback, early_stop, reduce_lr]
)

# --- STEP 7: FINAL EVALUATION & PICKLE EXPORT ---
print("\n--- Final Test Evaluation ---")
test_results = model.evaluate(test_gen)
print(f"Final Test Accuracy: {test_results[1]*100:.2f}%")

# Save the final trained brain as .pkl for your UI
with open('final_emotion_resnet.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nSuccess: Model saved as final_emotion_resnet.pkl")