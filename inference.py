# inference.py
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pickle

# ------------------------
# Paths
# ------------------------
model_path = 'models/face_recognition_model.pkl'
label_encoder_path = 'models/label_encoder.pkl'

# ------------------------
# Load trained classifier and label encoder
# ------------------------
with open(model_path, 'rb') as f:
    clf = pickle.load(f)

with open(label_encoder_path, 'rb') as f:
    le = pickle.load(f)

# ------------------------
# Initialize face detector and embedding model
# ------------------------
mtcnn = MTCNN(image_size=160, margin=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# ------------------------
# Function to predict actor from image
# ------------------------
def predict_actor(image_path):
    img = Image.open(image_path)
    img_cropped = mtcnn(img)
    if img_cropped is None:
        print("No face detected!")
        return
    embedding = resnet(img_cropped.unsqueeze(0))
    embedding_np = embedding.detach().numpy()
    pred_class = clf.predict(embedding_np)
    pred_prob = clf.predict_proba(embedding_np).max()
    actor_name = le.inverse_transform(pred_class)[0]
    print(f"Predicted Actor: {actor_name}, Confidence: {pred_prob*100:.2f}%")

# ------------------------
# Test on new image
# ------------------------
test_image = 'dataset/dhanush/img3.jpeg'  # replace with any image path
predict_actor(test_image)
