# exercise8_bugged.py
#
# Task:
#   3-class classification on 2D synthetic data, with train/val split.
#
# What you should debug:
#   - Why train accuracy gets high but val accuracy stays poor.
#   - Whether normalization is applied consistently.
#   - Whether train/val loops use the right modes (train/eval) and no_grad().
#   - Check shapes/dtypes/devices as usual.
#
# Hints:
#   - Focus on data pipeline symmetry between train and val.
#   - Check normalization, device, and loss/metric invariants.

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


NUM_CLASSES = 3


def make_data(n=900):
    """
    Make 3 Gaussian blobs in 2D:
      - class 0 around (-2, 0)
      - class 1 around (2, 0)
      - class 2 around (0, 2)
    """
    torch.manual_seed(0)
    n_per = n // NUM_CLASSES

    x0 = torch.randn(n_per, 2) * 0.5 + torch.tensor([-2.0, 0.0])
    x1 = torch.randn(n_per, 2) * 0.5 + torch.tensor([2.0, 0.0])
    x2 = torch.randn(n_per, 2) * 0.5 + torch.tensor([0.0, 2.0])

    X = torch.cat([x0, x1, x2], dim=0)
    y = torch.cat([
        torch.zeros(n_per),
        torch.ones(n_per),
        torch.full((n_per,), 2.0),
    ], dim=0)

    # shuffle
    idx = torch.randperm(X.size(0))
    return X[idx], y[idx].long()


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)  # logits: (batch, 3)


def train(
    n_samples=900,
    batch_size=64,
    epochs=10,
    lr=1e-2,
    val_frac=0.3,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    # ----- Make data -----
    X, y = make_data(n_samples)

    n_val = int(len(X) * val_frac)
    X_train, y_train = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    # ----- Normalization (BUGGY / ASYMMETRIC) -----
    # Compute mean/std on train only
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True) + 1e-6

    # Apply normalization to TRAIN
    X_train_norm = (X_train - mean) / std

    # BUG: val data left UNNORMALIZED
    X_val_norm = X_val  # <-- intentionally asymmetric

    train_ds = TensorDataset(X_train_norm, y_train)
    val_ds = TensorDataset(X_val_norm, y_val)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ----- Model / loss / opt -----
    model = SmallNet().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # ----- Train loop -----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            train_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)

        # ----- Validation loop (could be improved) -----
        # No eval() / no_grad() here on purpose for you to think about.
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        for xb, yb in val_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = crit(logits, yb)

            val_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            val_correct += (preds == yb).sum().item()
            val_total += yb.size(0)

        print(
            f"Epoch {epoch+1} "
            f"train_loss={train_loss/train_total:.4f} "
            f"train_acc={train_correct/train_total:.3f} | "
            f"val_loss={val_loss/val_total:.4f} "
            f"val_acc={val_correct/val_total:.3f}"
        )


if __name__ == "__main__":
    train()