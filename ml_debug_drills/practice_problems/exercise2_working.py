# exercise2_bugged.py
#
# INTENTIONALLY BUGGED:
# Simple CNN for classifying 8×8 synthetic “images” into 4 classes.
#
# Bugs to find & fix:
#   - Wrong input shape passed into Conv2d
#   - Missing flatten()
#   - Wrong final layer output dimension
#   - Labels as float instead of long
#   - Device mismatch
#   - Incorrect training loop order
#   - Accuracy computed incorrectly

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


NUM_CLASSES = 4


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # BUG: wrong output dimension (should be NUM_CLASSES)
        self.fc = nn.Linear(16 * 8 * 8, 3)

    def forward(self, x):
        # BUG: x is (batch, 64), but Conv2d needs (batch, 1, H, W)
        x = self.conv(x)
        # BUG: missing flatten
        logits = self.fc(x)
        return logits


def make_data(n_samples=400):
    torch.manual_seed(0)
    X = torch.randn(n_samples, 8, 8)   # shape (N, 8, 8)
    y = torch.randint(0, NUM_CLASSES, (n_samples,)).float()  # BUG: should be long, not float
    return X, y


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    X, y = make_data()
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    model = TinyCNN().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        correct = 0
        total = 0

        for xb, yb in dl:

            logits = model(xb)
            opt.zero_grad()

            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            # BUG: incorrect accuracy (thresholding instead of argmax)
            preds = (logits > 0.5).float().sum(dim=1)
            correct += (preds == yb).sum().item()

            total += yb.size(0)

        print(f"[BUGGED] Epoch {epoch+1}: acc={correct/total:.3f}")


if __name__ == "__main__":
    train()