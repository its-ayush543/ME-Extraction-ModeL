import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
import random

# --- Surrogate Architectures ---
class MLP_256(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, output_size=62):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.classifier(x)

class MLP_128(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, output_size=62):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

ARCHS = {
    "mlp_256": MLP_256,
    "mlp_128": MLP_128,
}

# --- Helper: Add label noise ---
def add_label_noise(labels, noise_level, num_classes):
    noisy_labels = labels.clone()
    n_samples = int(noise_level * len(labels))
    indices = torch.randperm(len(labels))[:n_samples]
    for i in indices:
        noisy_labels[i] = random.randint(0, num_classes - 1)
    return noisy_labels

# --- Load Data ---
embeddings = torch.load("stolen_embeddings.pt")
labels = torch.load("stolen_labels.pt")
num_classes = len(torch.unique(labels))

# --- Experiment Grid ---
query_sizes = [1000, 2000, 3000]
noise_levels = [0.0, 0.1, 0.2]
architectures = list(ARCHS.keys())

# --- Load target model ---
target_model = MLP_256().to("cpu")
target_model.load_state_dict(torch.load("models/target_model.pt", map_location="cpu"))
target_model.eval()

results = []
os.makedirs("checkpoints", exist_ok=True)

# --- Main Experiment Loop ---
for query_size in query_sizes:
    for noise in noise_levels:
        for arch_key in architectures:
            # Sample subset
            idx = torch.randperm(len(embeddings))[:query_size]
            X_sub = embeddings[idx]
            y_sub = labels[idx]

            # Add noise
            y_noisy = add_label_noise(y_sub, noise, num_classes)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_sub, y_noisy, test_size=0.2, stratify=y_noisy)

            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

            # Init model
            model = ARCHS[arch_key](output_size=num_classes)
            model.to("cpu")

            # Train
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            model.train()
            for epoch in range(10):
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()

            # Evaluate
            model.eval()
            y_pred = []
            y_true = []
            target_pred = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    logits = model(xb)
                    target_logits = target_model(xb)

                    pred = logits.argmax(dim=1)
                    target_pred.extend(target_logits.argmax(dim=1).tolist())
                    y_pred.extend(pred.tolist())
                    y_true.extend(yb.tolist())

            acc = accuracy_score(y_true, y_pred)
            fidelity = np.mean(np.array(y_pred) == np.array(target_pred))

            results.append({
                "query_size": query_size,
                "noise": noise,
                "arch": arch_key,
                "accuracy": acc,
                "fidelity": fidelity
            })

            # Save model
            torch.save(model.state_dict(), f"checkpoints/surrogate_{arch_key}_{query_size}_{int(noise*100)}.pt")

# Save all results
results_df = pd.DataFrame(results)
results_df.to_csv("results/surrogate_experiment_metrics.csv", index=False)
print("All experiments completed.")
