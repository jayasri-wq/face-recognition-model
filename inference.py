# inference.py
import tkinter as tk
from tkinter import filedialog
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import pickle
import numpy as np

# ------------------------
# Load models
# ------------------------
print("Loading model...")

# Load classifier
with open("models/face_recognition_model.pkl", "rb") as f:
    clf = pickle.load(f)

# Load label encoder
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Face detector
mtcnn = MTCNN(image_size=160, margin=20)

# Embedding model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

print("Model loaded successfully!")

# ------------------------
# Inference Function
# ------------------------
def predict_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    img = Image.open(file_path)

    # Detect face
    face = mtcnn(img)

    if face is None:
        result_label.config(text="No face detected. Try another image.")
        return

    # Generate embedding
    embedding = resnet(face.unsqueeze(0)).detach().numpy()

    # Predict class
    pred_class = clf.predict(embedding)[0]
    pred_proba = clf.predict_proba(embedding)[0]

    # Map class index to actor name
    actor_name = le.inverse_transform([pred_class])[0]

    confidence = np.max(pred_proba) * 100

    result_label.config(
        text=f"Predicted Actor: {actor_name}\nConfidence: {confidence:.2f}%"
    )

# ------------------------
# GUI Setup
# ------------------------
root = tk.Tk()
root.title("Face Recognition - Actor Identification")
root.geometry("400x200")

title_label = tk.Label(root, text="Actor Face Recognition", font=("Arial", 18))
title_label.pack(pady=10)

choose_btn = tk.Button(root, text="Choose Image", command=predict_image, font=("Arial", 14))
choose_btn.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
