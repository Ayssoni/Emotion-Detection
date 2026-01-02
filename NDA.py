import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import pickle

# --- STEP 1: DEFINE YOUR DATASET PATHS ---
# Replace these with your actual local folder paths
TRAIN_PATH = r'Dataset/basic/train'
TEST_PATH = r'/Users/aysoni/Documents/Emotion-Detection/Dataset/basic/test'

# --- STEP 2: SETUP DATA GENERATORS ---
# Training generator with Augmentation for accuracy
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2  # 20% of training data used for internal validation
)

# Testing generator (No Augmentation, only Rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# --- STEP 3: LOAD DATA FROM PATHS ---
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation data is part of the training cycle to tune parameters
val_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)

# Unseen data to check final performance
test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Do not shuffle to keep results consistent
)

# --- STEP 4: BUILD AND TRAIN THE MODEL ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax') # Happy, Sad, Angry
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# This trains the model
print("--- Starting Training ---")
model.fit(train_generator, validation_data=val_generator, epochs=15)

# --- STEP 5: EVALUATE ON TESTING DATASET ---
# This checks how well the model works on the unseen Test Path
print("--- Starting Final Evaluation ---")
scores = model.evaluate(test_generator)
print(f"Final Test Accuracy: {scores[1] * 100:.2f}%")

# --- STEP 6: SAVE FOR YOUR UI ---
with open('emotion_model.pkl', 'wb') as f:
    pickle.dump(model, f)