# mini_classifier_bugged.py
#
# INTENTIONALLY BUGGY VERSION of tiny 2D classifier.
# Candidate should debug:
# - label dtype for CrossEntropyLoss
# - device placement
# - output shape vs num_classes
# - gradient zeroing placement

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            # BUG 1: only 1 output, but using CrossEntropyLoss (expects 2 logits for 2 classes)
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def make_data(n_samples: int = 256):
    torch.manual_seed(0)
    x0 = torch.randn(n_samples // 2, 2) * 0.4 + torch.tensor([-1.0, -1.0])
    # BUG 2: labels are float instead of long
    y0 = torch.zeros(n_samples // 2)  # float by default

    x1 = torch.randn(n_samples // 2, 2) * 0.4 + torch.tensor([1.0, 1.0])
    y1 = torch.ones(n_samples // 2)   # float by default

    X = torch.cat([x0, x1], dim=0)
    y = torch.cat([y0, y1], dim=0)    # shape (N,), dtype float32
    return X, y


def train(num_epochs: int = 5, batch_size: int = 32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    X, y = make_data()
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = Net().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    xb0, yb0 = next(iter(dl))
    print("xb0:", xb0.shape, xb0.dtype, xb0.device)
    print("yb0:", yb0.shape, yb0.dtype, yb0.device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in dl:
            # BUG 3: not moving inputs/labels to same device as model
            # xb = xb.to(device)
            # yb = yb.to(device)

            # BUG 4: zero_grad after backward (wrong order)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.zero_grad()
            opt.step()

            running_loss += loss.item() * xb.size(0)

            # BUG 5: logits shape is (batch, 1); argmax over dim=1 acts weird,
            # and yb is float/different dtype/device.
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        print(f"[BUGGED] Epoch {epoch+1}: loss={running_loss/len(ds):.4f}, acc={correct/total:.3f}")


if __name__ == "__main__":
    train()