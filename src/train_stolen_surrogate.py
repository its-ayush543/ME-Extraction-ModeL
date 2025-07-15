# src/train_stolen_surrogate.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from target_model import SurrogateNet

# Load stolen data
X = torch.load("stolen_embeddings.pt")
y = torch.load("stolen_labels.pt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare data
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# Define model
model = SurrogateNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
print("[*] Training stolen surrogate model...")
for epoch in range(1, 16):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch}/15, Loss: {running_loss:.4f}")

# Evaluate
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = correct / total * 100
print(f"[+] Test Accuracy: {acc:.2f}%")

# Save model
torch.save(model.state_dict(), "models/stolen_surrogate_model.pt")
print("[+] Stolen model saved to 'models/stolen_surrogate_model.pt'")
