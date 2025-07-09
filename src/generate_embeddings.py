import os
import torch
from torchvision import datasets, transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
data_dir = 'lfw_faces'  # folder containing subfolders of each person
embedding_file = 'face_embeddings.pt'
label_file = 'face_labels.pt'

# Initialize MTCNN (for face detection + alignment) and InceptionResnetV1 (for embedding)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Prepare dataset
dataset = datasets.ImageFolder(data_dir)

# Store class names
class_names = dataset.classes

# Lists to store results
embedding_list = []
name_list = []

# Extract embeddings
print("Extracting embeddings...")

for img_path, label in tqdm(dataset.imgs):
    try:
        # Load image
        img = datasets.folder.default_loader(img_path)  # PIL Image

        # Detect and align face
        face_tensor = mtcnn(img)
        if face_tensor is not None:
            face_tensor = face_tensor.unsqueeze(0).to(device)  # Add batch dimension
            with torch.no_grad():
                embedding = model(face_tensor).squeeze().cpu()  # Remove batch, move to CPU
            embedding_list.append(embedding)
            name_list.append(class_names[label])
        else:
            print(f"[!] Face not detected in {img_path}")
    except Exception as e:
        print(f"[X] Error with {img_path}: {e}")

# Save results
print(f"Saving {len(embedding_list)} embeddings...")

torch.save(torch.stack(embedding_list), embedding_file)
torch.save(name_list, label_file)

print(f"Embeddings saved to '{embedding_file}'")
print(f"Labels saved to '{label_file}'")
