# exercise2_solution.py
#
# Fully working version of the tiny CNN classifier.

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
        self.fc = nn.Linear(16 * 8 * 8, NUM_CLASSES)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)


def make_data(n_samples=400):
    torch.manual_seed(0)
    X = torch.randn(n_samples, 1, 8, 8).float()
    y = torch.randint(0, NUM_CLASSES, (n_samples,)).long()
    return X, y


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    X, y = make_data()
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    model = TinyCNN().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        correct = 0
        total = 0

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        print(f"Epoch {epoch+1}: acc={correct/total:.3f}")


if __name__ == "__main__":
    train()
    