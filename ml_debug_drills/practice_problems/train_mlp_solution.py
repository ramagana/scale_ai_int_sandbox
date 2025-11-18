# train_mlp_solution.py
"""
Small, self-contained PyTorch example for debugging practice.
Trains a 2D → 2-class MLP on synthetic data.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_data(n_samples: int = 512):
    """
    Create a toy binary classification dataset in 2D.
    Label = 1 if x0 + x1 > 0, else 0.
    """
    torch.manual_seed(0)
    x = torch.randn(n_samples, 2)
    y = (x[:, 0] + x[:, 1] > 0).long()  # binary labels 0/1, correct dtype for CrossEntropy
    return x, y


class SimpleMLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=16, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train(num_epochs: int = 5, batch_size: int = 32, lr: float = 1e-2):
    x, y = make_data()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1}: loss={epoch_loss:.4f}")

    # quick sanity check
    with torch.no_grad():
        xb, yb = next(iter(loader))
        preds = model(xb).argmax(dim=1)
        acc = (preds == yb).float().mean().item()
        print(f"Sample batch accuracy: {acc:.3f}")


if __name__ == "__main__":
    train()