# src/simulate_attack.py

import torch
from target_model import TargetModel

# Load original embeddings
X = torch.load("embeddings/face_embeddings.pt")
y = torch.load("embeddings/face_labels.pt")

# Initialize the target model
target = TargetModel()

# Simulate black-box attack (query model for labels)
print("[*] Querying the target model...")
y_stolen = target.predict(X)

# Save stolen dataset
torch.save(X, "stolen_embeddings.pt")
torch.save(y_stolen, "stolen_labels.pt")

print("[+] Attack simulation complete. Stolen dataset saved.")

