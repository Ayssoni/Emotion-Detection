import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

# --- Theme Colors Extracted from Image ---
BG_COLOR = "#3D3945"          # Muted purple/gray sky
CARD_COLOR = "#2B2833"        # Deep shadow gray
ACCENT_GLOW = "#F9B382"       # Sunset orange/gold
TEXT_COLOR = "#D1CCD1"        # Light gray mist

class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Config
        self.title("Cinematic Emotion Detector")
        self.geometry("1100x650")
        self.configure(fg_color=BG_COLOR)
        
        self.is_live = False
        self.cap = None

        # Grid Setup
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Left Division (Video & Buttons)
        self.left_div = ctk.CTkFrame(self, fg_color=CARD_COLOR, corner_radius=20, border_width=1, border_color="#4A4655")
        self.left_div.grid(row=0, column=0, padx=30, pady=30, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.left_div, text="Awakening Camera...", text_color=TEXT_COLOR)
        self.video_label.pack(expand=True, fill="both", padx=20, pady=20)

        # Button Container
        self.button_div = ctk.CTkFrame(self.left_div, fg_color="transparent")
        self.button_div.pack(fill="x", pady=(0, 20))
        
        # Capture Button - Muted Style
        self.btn_capture = ctk.CTkButton(self.button_div, text="Capture", 
                                         fg_color="#5D576B", hover_color="#4A4655",
                                         text_color="white", corner_radius=10)
        self.btn_capture.pack(side="left", padx=30, expand=True)
        
        # Live Button - Glowing Orange Style
        self.btn_live = ctk.CTkButton(self.button_div, text="Go Live", command=self.toggle_live,
                                      fg_color=ACCENT_GLOW, hover_color="#E89E6D",
                                      text_color="#2B2833", font=("Arial", 14, "bold"), corner_radius=10)
        self.btn_live.pack(side="left", padx=30, expand=True)

        # 2. Right Division (Emotion Output)
        self.right_div = ctk.CTkFrame(self, fg_color=CARD_COLOR, corner_radius=20)
        self.right_div.grid(row=0, column=1, padx=(0, 30), pady=30, sticky="nsew")
        
        self.side_title = ctk.CTkLabel(self.right_div, text="EMOTION STATE", 
                                       font=("Serif", 14, "italic"), text_color=ACCENT_GLOW)
        self.side_title.pack(pady=40)
        
        # The main output area
        self.emotion_display = ctk.CTkLabel(self.right_div, text="...", 
                                            font=("Serif", 42, "bold"), text_color=TEXT_COLOR)
        self.emotion_display.pack(expand=True)

        # Start Camera with delay for macOS safety
        self.after(1000, self.init_camera)

    def init_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.update_video()
            else:
                self.video_label.configure(text="Camera Access Denied")
        except Exception as e:
            print(f"Error: {e}")

    def toggle_live(self):
        self.is_live = not self.is_live
        if self.is_live:
            self.btn_live.configure(text="Stop Live", fg_color="#E94560", text_color="white")
        else:
            self.btn_live.configure(text="Go Live", fg_color=ACCENT_GLOW, text_color="#2B2833")

    def update_video(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert to RGB for PIL
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                img_tk = ImageTk.PhotoImage(image=img.resize((650, 420)))
                
                self.video_label.configure(image=img_tk, text="")
                self.video_label.image = img_tk
                
                if self.is_live:
                    self.mock_ml_logic() # Replace with your model.predict()

        self.after(15, self.update_video)

    def mock_ml_logic(self):
        # This simulates your ML output
        import random
        emotions = ["VALOR", "STILLNESS", "CONFLICT", "GLORY"]
        self.emotion_display.configure(text=random.choice(emotions))

if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()