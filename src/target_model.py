import torch
import torch.nn as nn
import os

class SurrogateNet(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, output_size=62):
        super(SurrogateNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # classifier.0
            nn.ReLU(),                           # classifier.1
            nn.Dropout(0.3),                     # classifier.2
            nn.Linear(hidden_size, output_size)  # classifier.3
        )

    def forward(self, x):
        return self.classifier(x)

class TargetModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load trained model
        model_path = os.path.join("models", "surrogate_model.pt")
        self.model = SurrogateMLP(output_size=62).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, embeddings):
        with torch.no_grad():
            embeddings = embeddings.to(self.device)
            outputs = self.model(embeddings)
            predictions = torch.argmax(outputs, dim=1)
        return predictions
