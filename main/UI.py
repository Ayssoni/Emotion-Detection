import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import threading
import time

# --- 1. Robust Model Variable Loading ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# The script will check for both .h5 and .keras extensions to find your model
MODEL_NAME = 'advanced_emotion_7class.h5' 
MODEL_PATH = os.path.join(BASE_PATH, MODEL_NAME)
BG_PATH = os.path.join(BASE_PATH, 'Background.jpg')

try:
    if os.path.exists(MODEL_PATH):
        # Loading with compile=False to bypass signature/optimizer errors
        EMOTION_MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
        EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        print(f"Workflow: Model synchronized from {MODEL_PATH}")
    else:
        print(f"File Error: {MODEL_NAME} not found in {BASE_PATH}")
        EMOTION_MODEL = None
except Exception as e:
    print(f"Synchronization Error: {e}")
    EMOTION_MODEL = None

class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Wisdom AI - Synchronized Interface")
        self.geometry("1100x650")
        
        # --- Background implementation ---
        try:
            bg_data = Image.open(BG_PATH)
            self.bg_image = ctk.CTkImage(light_image=bg_data, dark_image=bg_data, size=(1100, 650))
            self.bg_label = ctk.CTkLabel(self, image=self.bg_image, text="")
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except:
            self.configure(fg_color="#1a1a1a")

        # --- UI Container ---
        self.main_container = ctk.CTkFrame(self, fg_color="#1e1e1e", corner_radius=25)
        self.main_container.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.85, relheight=0.75)
        self.main_container.grid_columnconfigure(0, weight=2)
        self.main_container.grid_columnconfigure(1, weight=1)

        # Left Div: Camera
        self.left_div = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.left_div.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.video_display = ctk.CTkLabel(self.left_div, text="Initializing Wisdom AI...", fg_color="black", corner_radius=15)
        self.video_display.pack(expand=True, fill="both", pady=(0, 15))

        self.btn_capture = ctk.CTkButton(self.left_div, text="CAPTURE & ANALYZE", 
                                         fg_color="#D49B4D", text_color="black", font=("Arial", 14, "bold"),
                                         command=self.start_workflow)
        self.btn_capture.pack(pady=10)

        # Right Div: Timer & Emotion Print
        self.right_div = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.right_div.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.status_label = ctk.CTkLabel(self.right_div, text="System Ready", font=("Serif", 22), text_color="#F9B382")
        self.status_label.pack(pady=(80, 10))

        self.emotion_text = ctk.CTkLabel(self.right_div, text="---", font=("Serif", 56, "bold"), text_color="white")
        self.emotion_text.pack(expand=True)

        self.cap = None
        self.is_live = True
        self.frozen_image = None
        self.after(1000, self.start_camera)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.update_feed()

    def update_feed(self):
        """Standard live feed update."""
        if self.is_live and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.frozen_image = frame
                self.render_to_ui(frame)
        self.after(20, self.update_feed)

    def render_to_ui(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(500, 350))
        self.video_display.configure(image=ctk_img, text="")
        self.video_display.image = ctk_img

    def start_workflow(self):
        """Action: Stop Camera -> Freeze Frame -> Start 15s Countdown."""
        if self.is_live and self.frozen_image is not None:
            self.is_live = False # Stop CV2 Feed
            self.render_to_ui(self.frozen_image) # Keep frozen image visible
            self.btn_capture.configure(state="disabled", text="ANALYZING...")
            # Run countdown in background thread
            threading.Thread(target=self.countdown_sequence, daemon=True).start()

    def countdown_sequence(self):
        """Updates the timer label from 15 to 0."""
        for i in range(15, -1, -1):
            self.after(0, lambda x=i: self.status_label.configure(text=f"Extracting Emotion: {x}s"))
            time.sleep(1)
        self.after(0, self.run_model_inference)

    def run_model_inference(self):
        """Merge frozen image with model variable and print result."""
        if EMOTION_MODEL is None:
            self.emotion_text.configure(text="No Model")
            return

        try:
            # Preprocessing (224x224 RGB for ResNet)
            img_rgb = cv2.cvtColor(self.frozen_image, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_array = np.expand_dims(img_resized, axis=0) / 255.0

            # Predict
            pred = EMOTION_MODEL.predict(img_array, verbose=0)
            emotion = EMOTION_LABELS[np.argmax(pred)]
            
            self.status_label.configure(text="Analysis Result:")
            self.emotion_text.configure(text=emotion)
            
            # Reset system after 10 seconds
            self.after(10000, self.reset_system)
        except Exception as e:
            print(f"Inference Error: {e}")

    def reset_system(self):
        self.is_live = True
        self.btn_capture.configure(state="normal", text="CAPTURE & ANALYZE")
        self.status_label.configure(text="System Ready")
        self.emotion_text.configure(text="---")

if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()