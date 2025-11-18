# exercise1_bugged.py
#
# INTENTIONALLY BUGGY:
# Train a 3-class classifier on synthetic 2D data.
#
# Fixes you should eventually make:
#   - label dtype for CrossEntropyLoss
#   - final layer output dimension
#   - device placement for xb/yb
#   - zero_grad() placement
#   - accuracy calculation

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


NUM_CLASSES = 3


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            # BUG 1: wrong output dimension (should be NUM_CLASSES)
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def make_data(n_samples: int = 300):
    """
    Make 3 clusters in 2D:
      - class 0 around (-1, -1)
      - class 1 around (0, 1)
      - class 2 around (1, -1)
    """
    torch.manual_seed(0)
    n = n_samples // 3

    x0 = torch.randn(n, 2) * 0.3 + torch.tensor([-1.0, -1.0])
    x1 = torch.randn(n, 2) * 0.3 + torch.tensor([0.0, 1.0])
    x2 = torch.randn(n, 2) * 0.3 + torch.tensor([1.0, -1.0])

    # BUG 2: labels are float by default (CE expects Long)
    y0 = torch.zeros(n)          # should be long
    y1 = torch.ones(n)           # should be long
    y2 = torch.full((n,), 2.0)   # should be long with value 2

    X = torch.cat([x0, x1, x2], dim=0)
    y = torch.cat([y0, y1, y2], dim=0)

    return X, y


def train(num_epochs: int = 5, batch_size: int = 32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    X, y = make_data()
    ds = TensorDataset(X, y)

    # BUG 3: no shuffling (not fatal, but suboptimal)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = Net().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    # One sanity batch
    xb0, yb0 = next(iter(dl))
    print("xb0:", xb0.shape, xb0.dtype, xb0.device)
    print("yb0:", yb0.shape, yb0.dtype, yb0.device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in dl:
            # BUG 4: xb and yb never moved to device
            # xb = xb.to(device)
            # yb = yb.to(device)

            logits = model(xb)

            # BUG 5: logits.shape will be (batch, 1), but we expect (batch, NUM_CLASSES).

            loss = crit(logits, yb)

            loss.backward()

            # BUG 6: zero_grad() after backward and before step (wipes grads)
            opt.zero_grad()
            opt.step()

            running_loss += loss.item() * xb.size(0)

            # BUG 7: accuracy computed with wrong logic; mixing shapes/dtypes
            # Softmax + threshold on one logit, comparing to float labels: all wrong.
            probs = logits.softmax(dim=1)  # shape (batch, 1)
            preds = (probs > 0.5).long().squeeze(-1)  # shape (batch,)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        print(f"[BUGGED] Epoch {epoch+1}: loss={running_loss/len(ds):.4f}, acc={correct/total:.3f}")


if __name__ == "__main__":
    train()