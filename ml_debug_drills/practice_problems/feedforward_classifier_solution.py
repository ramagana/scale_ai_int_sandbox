"""
feedforward_classifier_solution.py

Correct, working version of a small feedforward classifier on a synthetic 2D dataset.
Use this as reference after you attempt to fix feedforward_classifier_debug.py.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_synthetic_data(n_samples: int = 512):
    """
    Create a toy binary classification dataset in 2D.
    Class 0: centered near (-1, -1)
    Class 1: centered near (+1, +1)
    """
    torch.manual_seed(0)

    n_half = n_samples // 2

    x0 = torch.randn(n_half, 2) * 0.5 + torch.tensor([-1.0, -1.0])
    y0 = torch.zeros(n_half, dtype=torch.long)  # correct dtype: Long for CE

    x1 = torch.randn(n_half, 2) * 0.5 + torch.tensor([1.0, 1.0])
    y1 = torch.ones(n_half, dtype=torch.long)

    X = torch.cat([x0, x1], dim=0)
    y = torch.cat([y0, y1], dim=0)

    return X, y


class FeedforwardNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, num_classes=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),  # correct: two logits
        )

    def forward(self, x):
        return self.layers(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    X, y = make_synthetic_data(n_samples=512)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = FeedforwardNet(input_dim=2, hidden_dim=32, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)               # (batch, 2)
            loss = criterion(logits, yb)     # yb: (batch,), long

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

            preds = logits.argmax(dim=1)     # (batch,)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        epoch_loss = running_loss / len(dataset)
        acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, acc={acc:.3f}")

    # Quick eval example (no_grad, model in eval mode)
    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(loader))
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        acc = (preds == yb).float().mean().item()
        print(f"Sample batch eval accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()