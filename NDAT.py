import tensorflow as tf
import pickle
import numpy as np

# 1. Load Model
with open('advanced_emotion_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Point to the SPECIFIC image
img_path = '/Users/aysoni/Documents/Emotion-Detection/Dataset/basic/test/happy/im1.png'

# 3. Standard Loading (This handles the "No images found" error)
img = tf.keras.utils.load_img(img_path, target_size=(48, 48))
img_array = tf.keras.utils.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0) # This makes it a "batch" of 1

# 4. Predict
predictions = model.predict(img_array)
class_names = ['Angry', 'Happy', 'Sad']
print(f"Result: {class_names[np.argmax(predictions)]}")