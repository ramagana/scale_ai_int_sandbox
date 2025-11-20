"""
exercise4_bugged.py
INTENTIONALLY BUGGY

Task:
    Train a 4-class classifier on synthetic tabular data with 20 features.

Bugs to find:
    - Feature scaling applied incorrectly
    - One-hot labels passed into CrossEntropyLoss
    - Final layer output dim wrong
    - Softmax applied before CE
    - val split uses same data as train
    - Missing zero_grad
    - dtype mismatch: labels float
    - Device mismatch
    - Accuracy calculation broken
"""

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


NUM_CLASSES = 4
FEATURES = 20


def make_data(n_samples=400):
    torch.manual_seed(0)
    X = torch.randn(n_samples, FEATURES) * 10 + 50  # wide numeric range

    # class rule based on sum of feature blocks
    y = torch.argmin(
        torch.stack([
            X[:, :5].sum(dim=1),
            X[:, 5:10].sum(dim=1),
            X[:, 10:15].sum(dim=1),
            X[:, 15:].sum(dim=1),
        ], dim=1),
        dim=1,
    )

    # BUG: convert to one-hot float, CE expects int64
    y_oh = torch.nn.functional.one_hot(y, num_classes=NUM_CLASSES).float()

    return X.float(), y_oh  # wrong labels output


class BadClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 5),   # BUG: wrong num_classes
            nn.Softmax(dim=1),  # BUG: remove this for CE
        )

    def forward(self, x):
        return self.net(x)


def train(num_epochs=5, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    X, y = make_data()
    ds = TensorDataset(X, y)

    # BUG: train/val split incorrect – val is same data
    dl_train = DataLoader(ds, batch_size=batch_size, shuffle=False)
    dl_val = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = BadClassifier()  # BUG: never moved to device
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total = 0

        for xb, yb in dl_train:
            # BUG: no zero_grad()
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)  # BUG: logits already softmaxed
            loss = crit(logits, yb)  # BUG: CE expects class indices

            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)

            # BUG: accuracy using threshold on softmax for 4 classes
            preds = (logits > 0.5).float()
            correct = (preds == yb).all(dim=1).sum().item()
            total_correct += correct
            total += xb.size(0)

        print(f"[BUGGED] Epoch {epoch+1}: loss={total_loss/len(ds):.4f}, acc={total_correct/total:.3f}")


if __name__ == "__main__":
    train()