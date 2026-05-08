import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# -----------------------------------
# Dataset Class
# -----------------------------------

class PPGDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


# -----------------------------------
# CNN Model
# -----------------------------------

class CNNModel(nn.Module):

    def __init__(self):

        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=5
        )

        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=5
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(1088, 64)

        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.pool(self.relu(self.conv1(x)))

        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))

        x = self.fc2(x)

        return x


# -----------------------------------
# Load Dataset
# -----------------------------------

X = np.load("data/X.npy")

y = np.load("data/y.npy")

print("Dataset Loaded ✅")
print("X shape:", X.shape)
print("y shape:", y.shape)

dataset = PPGDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


# -----------------------------------
# Device Setup
# -----------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------------
# Initialize Model
# -----------------------------------

model = CNNModel().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------------
# Training Loop
# -----------------------------------

epochs = 500

for epoch in range(epochs):

    total_loss = 0

    for signals, labels in loader:

        signals = signals.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(signals).squeeze()

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {total_loss:.4f}"
    )


# -----------------------------------
# Save Model
# -----------------------------------

torch.save(
    model.state_dict(),
    "models/model.pth"
)

print("\nTraining Complete ✅")

print("Model Saved → models/model.pth")