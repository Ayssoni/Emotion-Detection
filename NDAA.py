import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import pickle

# --- STEP 1: PATH VARIABLES ---
TRAIN_PATH = r'/Users/aysoni/Documents/Emotion-Detection/Dataset/basic/train'
TEST_PATH = r'/Users/aysoni/Documents/Emotion-Detection/Dataset/basic/test'

# --- STEP 2: STABILIZED DATA GENERATORS (NO AUGMENTATION) ---
# We only rescale and set aside a validation split
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(TRAIN_PATH, target_size=(48,48), batch_size=64, class_mode='categorical', subset='training')
val_gen = datagen.flow_from_directory(TRAIN_PATH, target_size=(48,48), batch_size=64, class_mode='categorical', subset='validation')

# --- STEP 3: THE HIGH-ACCURACY CNN STRUCTURE ---
model = models.Sequential([
    # Block 1
    layers.Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 3)),
    layers.BatchNormalization(), # Normalizes the data for faster learning
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25), # Prevents memorization

    # Block 2
    layers.Conv2D(128, (5, 5), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(512, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# --- STEP 4: SMART CALLBACKS ---
# Stops training if accuracy stops improving to prevent "over-learning"
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Lowers the learning rate if the model gets "stuck"
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# --- STEP 5: TRAIN ---
model.fit(train_gen, validation_data=val_gen, epochs=50, callbacks=[early_stop, reduce_lr])

# Save the best model
with open('advanced_emotion_model.pkl', 'wb') as f:
    pickle.dump(model, f)