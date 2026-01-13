import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import os

# ==========================================
# 1. LOAD YOUR TRAINED MODEL HERE
# ==========================================
# Example: MY_MODEL = tensorflow.keras.models.load_model('my_model.h5')
MY_MODEL = None 

class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Wisdom AI - Emotion Detection")
        self.geometry("1100x600")
        
        # --- Background Image Logic ---
        # Ensure 'Background.jpg' is in your script folder
        bg_image_path = "Background.jpg"
        self.bg_image = Image.open(bg_image_path)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image.resize((1100, 600)))
        
        self.bg_label = ctk.CTkLabel(self, image=self.bg_photo, text="")
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # --- Overlay Main Container (Glassmorphism effect) ---
        self.main_container = ctk.CTkFrame(self, fg_color=("rgba(40, 40, 40, 0.7)"), corner_radius=15)
        self.main_container.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.85, relheight=0.7)

        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(1, weight=1)

        # --- Left Division: Video Feed ---
        self.left_div = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.left_div.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.video_display = ctk.CTkLabel(self.left_div, text="Initializing Wisdom AI...", 
                                          fg_color="black", corner_radius=10)
        self.video_display.pack(expand=True, fill="both", pady=(0, 10))

        # Buttons Sub-div
        self.btn_frame = ctk.CTkFrame(self.left_div, fg_color="transparent")
        self.btn_frame.pack(fill="x")
        
        self.btn_capture = ctk.CTkButton(self.btn_frame, text="Capture", 
                                         fg_color="#D49B4D", hover_color="#B3803D",
                                         text_color="black", font=("Arial", 12, "bold"),
                                         command=self.capture_and_predict)
        self.btn_capture.pack(side="left", padx=10, expand=True)

        self.btn_live = ctk.CTkButton(self.btn_frame, text="Go Live", 
                                      fg_color="#8FB94B", hover_color="#769C3A",
                                      text_color="black", font=("Arial", 12, "bold"),
                                      command=self.toggle_live)
        self.btn_live.pack(side="left", padx=10, expand=True)

        # --- Right Division: Output ---
        self.right_div = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.right_div.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.out_header = ctk.CTkLabel(self.right_div, text="Detected Emotion", 
                                       font=("Serif", 24), text_color="#F9B382")
        self.out_header.pack(pady=(40, 20))

        self.emotion_result = ctk.CTkLabel(self.right_div, text="---", 
                                           font=("Serif", 48, "bold"), text_color="white")
        self.emotion_result.pack(expand=True)

        # Camera setup
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
                # Mirror and convert for display
                frame = cv2.flip(frame, 1)
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_tk = ImageTk.PhotoImage(image=img.resize((400, 280)))
                
                self.video_display.configure(image=img_tk, text="")
                self.video_display.image = img_tk

                if self.is_live:
                    self.predict_emotion(frame)

        self.after(20, self.update_frame)

    def toggle_live(self):
        self.is_live = not self.is_live
        self.btn_live.configure(text="Stop Live" if self.is_live else "Go Live",
                                 fg_color="#E94560" if self.is_live else "#8FB94B")

    def capture_and_predict(self):
        ret, frame = self.cap.read()
        if ret:
            self.predict_emotion(frame)

    def predict_emotion(self, frame):
        """
        Connects the UI to your variable MY_MODEL
        """
        if MY_MODEL is None:
            # Placeholder if no model is loaded yet
            results = ["Focused", "Determined", "Calm", "Wise"]
            import random
            prediction = random.choice(results)
        else:
            # 1. Preprocess frame (resize, grayscale, etc. as per your model)
            # processed_frame = preprocess(frame) 
            # 2. Run prediction
            # prediction = MY_MODEL.predict(processed_frame)
            prediction = "Model Loaded" 

        self.emotion_result.configure(text=prediction)

if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()