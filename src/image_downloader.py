from sklearn.datasets import fetch_lfw_people
import numpy as np
import os
from PIL import Image

# Fetch the dataset in grayscale or color
lfw = fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True)

print(f"Downloaded {len(lfw.images)} images of {len(lfw.target_names)} people")
