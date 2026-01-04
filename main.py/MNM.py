import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import pickle
import ssl
import os
os.environ['PYTHONHTTPSVERIFY'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context

# --- STEP 1: DEFINE DATASET PATH VARIABLES ---
TRAIN_DIR = r'Dataset/train'
TEST_DIR = r'Dataset/test'

# --- STEP 2: ADVANCED DATA AUGMENTATION ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load 7 classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(48, 48), batch_size=64, class_mode='categorical', subset='training'
)
val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(48, 48), batch_size=64, class_mode='categorical', subset='validation'
)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(48, 48), batch_size=32, class_mode='categorical', shuffle=False
)

# --- STEP 3: TRANSFER LEARNING (MOBILENETV2) ---
# MobileNetV2 is efficient for 48x48 images 
base_model = MobileNetV2(input_shape=(48, 48, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax') # 7 universal emotions
])

# --- STEP 4: STAGE 1 TRAINING (TOP LAYERS ONLY) ---
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- Phase 1: Training Top Layers ---")
model.fit(train_gen, validation_data=val_gen, epochs=10)

# --- STEP 5: PHASE 2 FINE-TUNING (UNFREEZE BASE) ---
# Unfreeze the last 20 layers of the base model for precision
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with a MUCH lower learning rate for fine-tuning
model.compile(optimizer=optimizers.Adam(learning_rate=0.00001), 
              loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

print("\n--- Phase 2: Fine-Tuning the Model ---")
model.fit(train_gen, validation_data=val_gen, epochs=30, callbacks=[early_stop])

# --- STEP 6: EVALUATE & SAVE AS PKL ---
results = model.evaluate(test_gen)
print(f"Final Test Accuracy: {results[1]*100:.2f}%")

# Save as .pkl (Note: .keras is generally safer, but pkl works for your requirement)
with open('advanced_emotion_7class.h5', 'wb') as f:
    pickle.dump(model, f)
print("Model successfully saved as advanced_emotion_7class.pkl")