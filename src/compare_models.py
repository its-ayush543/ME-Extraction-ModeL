import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import os
import random

class SurrogateMLP(torch.nn.Module):
    def __init__(self, input_size=512, hidden_size=256, output_size=62):
        super(SurrogateMLP, self).__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
embeddings = torch.load("stolen_embeddings.pt").to(device)
labels = torch.load("stolen_labels.pt").to(device)

# Full dataset
dataset = TensorDataset(embeddings, labels)

# Load models
target_model = SurrogateMLP(output_size=62).to(device)
stolen_model = SurrogateMLP(output_size=62).to(device)

target_model.load_state_dict(torch.load("models/target_model.pt", map_location=device))
stolen_model.load_state_dict(torch.load("models/surrogate_model.pt", map_location=device))

target_model.eval()
stolen_model.eval()

# Sample sizes (percentages of total dataset)
sample_percents = [0.1, 0.3, 0.5, 0.7, 1.0]
accuracies = []
fidelities = []

for percent in sample_percents:
    size = int(len(dataset) * percent)
    indices = random.sample(range(len(dataset)), size)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=64, shuffle=False)

    true_labels = []
    target_preds = []
    stolen_preds = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            true_labels.extend(y.cpu().numpy())

            t_logits = target_model(x)
            s_logits = stolen_model(x)

            t_pred = torch.argmax(t_logits, dim=1)
            s_pred = torch.argmax(s_logits, dim=1)

            target_preds.extend(t_pred.cpu().numpy())
            stolen_preds.extend(s_pred.cpu().numpy())

    y_true = np.array(true_labels)
    t_pred = np.array(target_preds)
    s_pred = np.array(stolen_preds)

    accuracy = accuracy_score(y_true, s_pred)
    fidelity = np.mean(t_pred == s_pred)

    accuracies.append(accuracy)
    fidelities.append(fidelity)

    print(f"[{int(percent * 100)}% Data] Accuracy: {accuracy:.4f}, Fidelity: {fidelity:.4f}")

# Plotting
plt.figure(figsize=(7, 5))
plt.plot(fidelities, accuracies, marker='o', linestyle='-', color='blue')
for i, p in enumerate(sample_percents):
    plt.text(fidelities[i], accuracies[i], f"{int(p*100)}%", fontsize=9, ha='right')

plt.title("Accuracy vs Fidelity (varying data size)")
plt.xlabel("Fidelity")
plt.ylabel("Accuracy")
plt.grid(True)
os.makedirs("results", exist_ok=True)
plt.savefig("results/accuracy_vs_fidelity_multiple.png")
plt.show()

# Optional: Save results to CSV
import pandas as pd
df = pd.DataFrame({
    "SampleSizePercent": [int(p*100) for p in sample_percents],
    "Accuracy": accuracies,
    "Fidelity": fidelities
})
df.to_csv("results/accuracy_vs_fidelity_multiple.csv", index=False)
