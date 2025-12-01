# train.py
import os
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# ------------------------
# Paths
# ------------------------
dataset_path = 'dataset'                  # folder with actor images
model_save_path = 'models/face_recognition_model.pkl'  # save as pickle
label_encoder_path = 'models/label_encoder.pkl'

# ------------------------
# Initialize face detector and embedding model
# ------------------------
mtcnn = MTCNN(image_size=160, margin=20)               # detect and crop faces
resnet = InceptionResnetV1(pretrained='vggface2').eval()  # embedding model

# ------------------------
# Prepare dataset
# ------------------------
X = []  # embeddings
y = []  # labels

print("Processing images...")
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        try:
            img = Image.open(img_path)
            img_cropped = mtcnn(img)
            if img_cropped is not None:
                embedding = resnet(img_cropped.unsqueeze(0))
                X.append(embedding.detach().numpy()[0])
                y.append(person_name)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

X = np.array(X)
y = np.array(y)

print(f"Total samples processed: {len(y)}")

# ------------------------
# Encode labels
# ------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save label encoder
os.makedirs('models', exist_ok=True)
with open(label_encoder_path, 'wb') as f:
    pickle.dump(le, f)

# ------------------------
# Train classifier
# ------------------------
print("Training classifier...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y_encoded)

# Save classifier using pickle
with open(model_save_path, 'wb') as f:
    pickle.dump(clf, f)

print(f"Model saved to {model_save_path}")
print("Training complete!")
