import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

# ==========================================
# 1. LOAD YOUR MODEL
# ==========================================
# Ensure 'my_model.keras' is in the same folder as this script
try:
    MY_MODEL = tf.keras.models.load_model('main/emotion_resnet_best.keras')
    # Define your emotion labels in the order your model was trained
    EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
except Exception as e:
    print(f"Error loading model: {e}")
    MY_MODEL = None
    EMOTION_LABELS = ["Error"]

class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Wisdom AI - Emotion Detection")
        self.geometry("1100x650")
        
        # --- Background Image ---
        bg_image_path = "Background.jpg"
        self.bg_image = Image.open(bg_image_path)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image.resize((1100, 650)))
        
        self.bg_label = ctk.CTkLabel(self, image=self.bg_photo, text="")
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # --- Glassmorphism Container ---
        self.main_container = ctk.CTkFrame(self, fg_color="rgba(30, 30, 30, 0.6)", corner_radius=20)
        self.main_container.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.85, relheight=0.75)

        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(1, weight=1)

        # --- Left Side: Video ---
        self.left_div = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.left_div.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.video_display = ctk.CTkLabel(self.left_div, text="Connecting to Camera...", 
                                          fg_color="black", corner_radius=15)
        self.video_display.pack(expand=True, fill="both", pady=(0, 15))

        self.btn_frame = ctk.CTkFrame(self.left_div, fg_color="transparent")
        self.btn_frame.pack(fill="x")
        
        self.btn_capture = ctk.CTkButton(self.btn_frame, text="Capture", 
                                         fg_color="#D49B4D", hover_color="#B3803D",
                                         text_color="black", font=("Arial", 14, "bold"),
                                         command=self.capture_and_predict)
        self.btn_capture.pack(side="left", padx=10, expand=True)

        self.btn_live = ctk.CTkButton(self.btn_frame, text="Go Live", 
                                      fg_color="#8FB94B", hover_color="#769C3A",
                                      text_color="black", font=("Arial", 14, "bold"),
                                      command=self.toggle_live)
        self.btn_live.pack(side="left", padx=10, expand=True)

        # --- Right Side: Output ---
        self.right_div = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.right_div.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.out_header = ctk.CTkLabel(self.right_div, text="Detected Emotion", 
                                       font=("Serif", 28), text_color="#F9B382")
        self.out_header.pack(pady=(50, 20))

        self.emotion_result = ctk.CTkLabel(self.right_div, text="---", 
                                           font=("Serif", 54, "bold"), text_color="white")
        self.emotion_result.pack(expand=True)

        # Camera state
        self.is_live = False
        self.cap = None
        self.after(1000, self.start_camera)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_tk = ImageTk.PhotoImage(image=img.resize((450, 320)))
                
                self.video_display.configure(image=img_tk, text="")
                self.video_display.image = img_tk

                if self.is_live:
                    self.predict_emotion(frame)

        self.after(30, self.update_frame)

    def toggle_live(self):
        self.is_live = not self.is_live
        self.btn_live.configure(text="Stop Live" if self.is_live else "Go Live",
                                 fg_color="#E94560" if self.is_live else "#8FB94B")

    def capture_and_predict(self):
        ret, frame = self.cap.read()
        if ret:
            self.predict_emotion(frame)

    def predict_emotion(self, frame):
        if MY_MODEL is None:
            self.emotion_result.configure(text="No Model")
            return

        try:
            # 1. Preprocessing
            # Convert to grayscale (common for emotion models)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize to model input size (usually 48x48)
            resized = cv2.resize(gray, (48, 48))
            # Normalize pixel values to [0, 1]
            normalized = resized / 255.0
            # Reshape for Keras (Batch, Height, Width, Channels)
            reshaped = np.reshape(normalized, (1, 48, 48, 1))

            # 2. Inference
            prediction = MY_MODEL.predict(reshaped, verbose=0)
            max_index = int(np.argmax(prediction))
            final_emotion = EMOTION_LABELS[max_index]

            # 3. Update UI
            self.emotion_result.configure(text=final_emotion)
        except Exception as e:
            print(f"Prediction Error: {e}")

if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()