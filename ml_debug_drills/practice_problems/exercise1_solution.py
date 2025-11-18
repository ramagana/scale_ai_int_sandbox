# exercise1_solution.py
#
# Clean, fixed version of the 3-class classifier.

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
            # FIX 1: output dimension = number of classes
            nn.Linear(16, NUM_CLASSES),
        )

    def forward(self, x):
        return self.layers(x)


def make_data(n_samples: int = 300):
    torch.manual_seed(0)
    n = n_samples // 3

    x0 = torch.randn(n, 2) * 0.3 + torch.tensor([-1.0, -1.0])
    x1 = torch.randn(n, 2) * 0.3 + torch.tensor([0.0, 1.0])
    x2 = torch.randn(n, 2) * 0.3 + torch.tensor([1.0, -1.0])

    # FIX 2: labels as Long class indices
    y0 = torch.zeros(n, dtype=torch.long)
    y1 = torch.ones(n, dtype=torch.long)
    y2 = torch.full((n,), 2, dtype=torch.long)

    X = torch.cat([x0, x1, x2], dim=0)  # (N, 2)
    y = torch.cat([y0, y1, y2], dim=0)  # (N,)

    return X, y


def train(num_epochs: int = 5, batch_size: int = 32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    X, y = make_data()
    ds = TensorDataset(X, y)

    # FIX 3: shuffle during training
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = Net().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    # One-time sanity check
    xb0, yb0 = next(iter(dl))
    print("xb0:", xb0.shape, xb0.dtype, xb0.device)
    print("yb0:", yb0.shape, yb0.dtype, yb0.device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            # FIX 4: correct order: zero_grad → forward → loss → backward → step
            opt.zero_grad()

            logits = model(xb)           # (batch, NUM_CLASSES)
            loss = crit(logits, yb)

            loss.backward()
            opt.step()

            running_loss += loss.item() * xb.size(0)

            # FIX 5: accuracy via argmax over class dimension
            preds = logits.argmax(dim=1)  # (batch,)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        print(f"[SOLUTION] Epoch {epoch+1}: loss={running_loss/len(ds):.4f}, acc={correct/total:.3f}")


if __name__ == "__main__":
    train()