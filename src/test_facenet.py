import os
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from numpy import expand_dims
from sklearn.preprocessing import Normalizer
from mtcnn.mtcnn import MTCNN
from PIL import Image

# Initialize MTCNN for face detection
detector = MTCNN()

# Load the pre-trained FaceNet model (you must have it downloaded as .h5)
model = load_model('facenet_keras.h5')
print("FaceNet model loaded.")

# Helper: Preprocess face
def preprocess_face(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    return expand_dims(face, axis=0)

# Helper: Extract face from image using MTCNN
def extract_face(file_path):
    img = cv2.imread(file_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    if results:
        x1, y1, width, height = results[0]['box']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = x1 + width, y1 + height
        face = img_rgb[y1:y2, x1:x2]
        return face
    return None

# L2 Normalizer
l2_normalizer = Normalizer('l2')

# Folder containing all person subfolders
data_folder = 'lfw_faces'  # Change if using a different folder
output_data = []

for person in os.listdir(data_folder):
    person_dir = os.path.join(data_folder, person)
    if not os.path.isdir(person_dir):
        continue
    for file in os.listdir(person_dir):
        file_path = os.path.join(person_dir, file)
        face = extract_face(file_path)
        if face is None:
            print(f"Face not detected in {file}")
            continue
        face_array = preprocess_face(face)
        embedding = model.predict(face_array)[0]
        embedding = l2_normalizer.transform([embedding])[0]
        output_data.append([person] + embedding.tolist())

# Save embeddings to CSV
df = pd.DataFrame(output_data)
df.to_csv("embeddings_log.csv", index=False)
print("Embeddings saved to embeddings_log.csv.")
