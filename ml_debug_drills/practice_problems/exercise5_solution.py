"""
exercise5_solution.py
Correct implementation for Exercise 5.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_data(n=300):
    torch.manual_seed(0)
    X = torch.randn(n, 6) * 20 + 100
    y = (X[:, 0] + X[:, 3] > 100).long()  # convert to class index

    return X.float(), y


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # logits for 2 classes
        )

    def forward(self, x):
        return self.net(x)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ----- Data -----
    X, y = make_data()

    # Correct normalization: across features (dim=0)
    X_norm = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-6)

    ds = TensorDataset(X_norm, y)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    # ----- Model / loss / optimizer -----
    model = Classifier().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)

    for epoch in range(3):
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()

            logits = model(xb)            # (batch, 2)
            loss = crit(logits, yb)       # CE with long labels

            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)  # (batch,)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        print(
            f"Epoch {epoch+1}: loss={total_loss/len(ds):.4f}, acc={correct/total:.3f}"
        )


if __name__ == "__main__":
    train()