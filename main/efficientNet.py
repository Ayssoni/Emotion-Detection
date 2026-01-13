import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
IMG_SIZE = 224 # Upscaling FER images (48x48) to 224x224 for EfficientNet
BATCH_SIZE = 32
NUM_CLASSES = 7
EPOCHS = 20 # Increase to 50+ for maximum accuracy
TRAIN_DIR = 'Dataset/FER/train'
TEST_DIR = 'Dataset/FER/train'

# --- 2. DATA PREPARATION & AUGMENTATION ---
# Augmentation is critical to prevent overfitting on small facial datasets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Keep False for testing/evaluation
)

# --- 3. MODEL BUILDING (Transfer Learning) ---
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False # Start by freezing pre-trained weights

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5), # High dropout for small face datasets
    layers.Dense(512, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 4. CALLBACKS FOR AUTO-SAVING ---
# This ensures we save only the BEST version seen during training
checkpoint = ModelCheckpoint(
    'best_emotion_model.h5', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

# Prevents training from wasting time if accuracy stops improving
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Lowers learning rate if the model hits a plateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# --- 5. TRAINING ---
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# --- 6. TESTING & FINAL SAVE ---
print("\n--- Evaluating on Test Data ---")
loss, accuracy = model.evaluate(test_generator)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")

# Explicitly save the final trained model
model.save('final_emotion_model.h5')
print("Model saved as 'final_emotion_model.h5'")