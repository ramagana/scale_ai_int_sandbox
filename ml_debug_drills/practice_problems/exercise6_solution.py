"""
exercise6_solution.py

Binary classification on synthetic tabular data using BCEWithLogitsLoss.
Correct:
    - label shapes and dtypes
    - loss function
    - device placement
    - training loop
    - accuracy computation
"""

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


INPUT_DIM = 10


def make_data(n_samples: int = 300):
    torch.manual_seed(0)
    X = torch.randn(n_samples, INPUT_DIM)

    scores = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * (X[:, 2] * X[:, 3])
    y = (scores > 0).float()  # (N,)

    # For BCEWithLogitsLoss, target shape can be (N,) or (N,1); here we keep (N,1) consistently.
    y = y.unsqueeze(1)  # (N,1)

    return X.float(), y.float()


class BinaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # logits
        )

    def forward(self, x):
        return self.net(x)  # (N,1)


def train(num_epochs: int = 5, batch_size: int = 32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    X, y = make_data()
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = BinaryNet().to(device)
    crit = nn.BCEWithLogitsLoss()  # expects logits + float targets in [0,1]
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(num_epochs):
        total_loss = 0.0
        total = 0
        correct = 0

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()

            logits = model(xb)              # (N,1)
            loss = crit(logits, yb)         # logits + float targets

            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)

            # predictions: sigmoid(logits) > 0.5
            probs = torch.sigmoid(logits)   # (N,1)
            preds = (probs > 0.5).float()   # (N,1)

            correct += (preds == yb).sum().item()
            total += yb.numel()

        avg_loss = total_loss / len(ds)
        acc = correct / total
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.3f}")


if __name__ == "__main__":
    train()