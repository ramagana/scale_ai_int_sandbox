"""
exercise6_bugged.py
INTENTIONALLY BUGGY

Task:
    Binary classification on synthetic tabular data (spam / not-spam style).

Bugs to find:
    - wrong loss (MSE instead of BCE/BCEWithLogits)
    - manual sigmoid + BCELoss misuse
    - wrong label dtype (float vs long)
    - wrong target shape (N,1 vs N)
    - missing zero_grad()
    - accuracy logic incorrect
    - device issues
"""

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


INPUT_DIM = 10


def make_data(n_samples: int = 300):
    torch.manual_seed(0)
    X = torch.randn(n_samples, INPUT_DIM)

    # non-linear rule for binary label
    scores = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * (X[:, 2] * X[:, 3])
    y = (scores > 0).float()  # 0.0 or 1.0

    # BUG 1: return y as shape (N, 1) instead of (N,)
    y = y.unsqueeze(1)  # (N, 1)

    return X, y


class BinaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # logits (scalar per sample)
        )

    def forward(self, x):
        return self.net(x)  # (N, 1)


def train(num_epochs: int = 5, batch_size: int = 32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    X, y = make_data()
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)  # BUG 2: no shuffle

    model = BinaryNet()  # BUG 3: never to(device)

    # BUG 4: use plain MSELoss instead of BCEWithLogits or BCELoss
    crit = nn.MSELoss()

    # BUG 5: large LR, can cause instability
    opt = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        total_loss = 0.0
        total = 0
        correct = 0

        for xb, yb in dl:
            # BUG 6: move only inputs to device, not labels
            xb = xb.to(device)

            # BUG 7: missing optimizer.zero_grad()
            logits = model(xb)      # (N,1)

            # BUG 8: apply sigmoid and still use MSELoss on probs
            probs = torch.sigmoid(logits)   # (N,1)

            loss = crit(probs, yb)          # yb on CPU, shape mismatch semantics

            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)

            # BUG 9: accuracy: comparing float probs > 0.0 to float labels (N,1)
            preds = (probs > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.numel()

        print(f"[BUGGED] Epoch {epoch+1}: loss={total_loss/len(ds):.4f}, acc={correct/total:.3f}")


if __name__ == "__main__":
    train()