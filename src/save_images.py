from sklearn.datasets import fetch_lfw_people
import os
from PIL import Image

# Load LFW again
lfw = fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True)

print(f"Saving {len(lfw.images)} images...")

# Create output folder
os.makedirs("lfw_faces", exist_ok=True)

# Save each image
for i, image in enumerate(lfw.images):
    label = lfw.target_names[lfw.target[i]].replace(" ", "_")
    person_dir = os.path.join("lfw_faces", label)
    os.makedirs(person_dir, exist_ok=True)
    
    img = Image.fromarray((image * 255).astype('uint8'))  # convert from float to uint8
    path = os.path.join(person_dir, f"{label}_{i}.jpg")
    img.save(path)

print("✅ Images saved to ./lfw_faces/")
