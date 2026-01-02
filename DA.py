import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import os

# --- STEP 1: PATH VARIABLES ---
# Ensure these folders contain 'Happy', 'Sad', and 'Angry' sub-folders
TRAIN_PATH = r'/Users/aysoni/Documents/Emotion-Detection/Dataset/basic/train'
TEST_PATH = r'/Users/aysoni/Documents/Emotion-Detection/Dataset/basic/test'

# --- STEP 2: DATA GENERATORS ---
# Training with Augmentation to make the model robust for your UI
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Internal check during training
)

# Testing (Strictly NO augmentation, only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# --- STEP 3: LOAD DATA ---
train_gen = train_datagen.flow_from_directory(
    TRAIN_PATH, target_size=(48,48), batch_size=64, class_mode='categorical', subset='training'
)
val_gen = train_datagen.flow_from_directory(
    TRAIN_PATH, target_size=(48,48), batch_size=64, class_mode='categorical', subset='validation'
)
test_gen = test_datagen.flow_from_directory(
    TEST_PATH, target_size=(48,48), batch_size=32, class_mode='categorical', shuffle=False
)

# --- STEP 4: ADVANCED ARCHITECTURE ---
model = models.Sequential([
    layers.Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(512, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# --- STEP 5: SMART TRAINING ---
# restore_best_weights ensures we keep the most accurate version
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

print("\n--- Starting Master Training with Augmentation ---")
history = model.fit(train_gen, validation_data=val_gen, epochs=100, callbacks=[early_stop, reduce_lr])

# --- STEP 6: FINAL TESTING / EVALUATION ---
print("\n--- Starting Final Evaluation on Unseen Test Dataset ---")
test_results = model.evaluate(test_gen)
print(f"\nFINAL TEST ACCURACY: {test_results[1] * 100:.2f}%")
print(f"FINAL TEST LOSS: {test_results[0]:.4f}")

# --- STEP 7: SAVE EVERYTHING ---
with open('master_emotion_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved successfully as 'master_emotion_model.pkl'")