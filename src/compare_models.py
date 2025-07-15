import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from target_model import SurrogateNet
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# Load test data
embeddings = torch.load("embeddings/face_embeddings.pt")
labels = torch.load("embeddings/face_labels.pt")

# Encode labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = torch.tensor(label_encoder.fit_transform(labels))

# Train-test split
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(embeddings, labels_encoded, test_size=0.2, stratify=labels_encoded)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper to load and evaluate a model
def evaluate_model(model_path, model_name):
    model = SurrogateNet(output_size=len(label_encoder.classes_)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, cm

# Paths
original_path = os.path.join("models", "surrogate_model.pt")
stolen_path = os.path.join("models", "stolen_surrogate_model.pt")

# Evaluate both models
original_acc, original_cm = evaluate_model(original_path, "Original")
stolen_acc, stolen_cm = evaluate_model(stolen_path, "Stolen")

# Print accuracy
print(f"Original Model Accuracy: {original_acc * 100:.2f}%")
print(f"Stolen Model Accuracy: {stolen_acc * 100:.2f}%")

# Bar chart comparison
plt.figure(figsize=(6, 4))
plt.bar(["Original", "Stolen"], [original_acc * 100, stolen_acc * 100], color=["blue", "red"])
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 100)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("comparison_accuracy.png")
plt.show()

# Optional: Confusion matrices
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(original_cm, cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)
plt.title("Original Model Confusion Matrix")

plt.subplot(1, 2, 2)
sns.heatmap(stolen_cm, cmap="Reds", cbar=False, xticklabels=False, yticklabels=False)
plt.title("Stolen Model Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.show()
