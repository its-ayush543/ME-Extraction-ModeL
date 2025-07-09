import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# Load embeddings and labels
embeddings = torch.load("face_embeddings.pt")
labels = torch.load("face_labels.pt")

# Encode labels numerically
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = torch.tensor(label_encoder.fit_transform(labels))

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels_encoded, test_size=0.2, stratify=labels_encoded)

# Create DataLoaders
train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# Define MLP surrogate model
class SurrogateMLP(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, output_size=62):
        super(SurrogateMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.classifier(x)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SurrogateMLP(output_size=len(label_encoder.classes_)).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# Evaluate
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Save model
torch.save(model.state_dict(), "surrogate_model.pt")
print("Model saved as surrogate_model.pt")
