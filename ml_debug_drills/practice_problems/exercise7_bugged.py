# exercise7_bugged.py
# INTENTIONALLY BUGGY
#
# Task:
#   Binary classification on 50-feature tabular data.
#   Ground truth uses only 3 features.
#
# Bugs:
#   1. Label encoding uses { -1, +1 } but BCE expects {0,1}
#   2. Feature normalization uses dim=1 instead of dim=0
#   3. Model final layer outputs shape (batch, 2) but BCE expects (batch,) or (batch,1)
#   4. Missing sigmoid — or wrong loss (BCE vs BCEWithLogitsLoss)
#   5. Accuracy threshold uses > 0.5 on logits instead of sigmoid(logits)
#   6. shuffle=False for training DataLoader
#   7. xb and yb never moved to device
# --------------------------------------------------------------

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def make_data(n=500):
    torch.manual_seed(0)

    X = torch.randn(n, 50)

    # Ground truth uses only 3 features
    y_raw = X[:, 3] * 0.8 + X[:, 7] * -1.2 + X[:, 12] * 0.5

    # BUG 1: encode labels as -1/+1 (bad for BCE)
    y = torch.where(y_raw > 0, -1.0, 1.0)

    return X, y


class BadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # BUG 3: BCE expects one output
        )

    def forward(self, x):
        return self.net(x)  # logits shape (batch,2) WRONG


def train(epochs=5, batch_size=32, lr=1e-2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    X, y = make_data(500)

    # BUG 2: normalize across dim=1 (per sample), removes true signal
    X = (X - X.mean(dim=1, keepdim=True)) / (X.std(dim=1, keepdim=True) + 1e-6)

    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)  # BUG 6

    model = BadNet().to(device)

    # BUG 4: BCE (expects probabilities OR BCEWithLogitsLoss with raw logits)
    crit = nn.BCELoss()

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_correct = 0
        total = 0

        for xb, yb in dl:

            # BUG 7: no .to(device)
            xb = xb
            yb = yb

            opt.zero_grad()

            logits = model(xb)

            # logits shape (batch,2) → BCE expects (batch) or (batch,1)
            # yb shape (batch,)
            # This silently triggers broadcasting...
            loss = crit(logits, yb)

            loss.backward()
            opt.step()

            epoch_loss += loss.item() * xb.size(0)

            # BUG 5: accuracy computed from logits directly
            preds = (logits > 0.5).float()

            correct = (preds.squeeze() == yb).sum().item()
            epoch_correct += correct
            total += xb.size(0)

        print(f"[BUGGED] Epoch {epoch+1} loss={epoch_loss/total:.4f}, acc={epoch_correct/total:.3f}")


if __name__ == "__main__":
    train()