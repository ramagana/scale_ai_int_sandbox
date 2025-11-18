# exercise3_bugged.py
#
# INTENTIONALLY BUGGED:
# Tiny MLP to fit y = 2x + 3 + noise (1-D regression).
#
# Bugs to find & fix:
#   - Wrong input shape to Linear (x is 1D instead of 2D)
#   - Dtype issues (default float64)
#   - No shuffle in DataLoader
#   - Model never moved to device
#   - Missing zero_grad()
#   - Too-large learning rate causes unstable training

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class TinyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_data(n_samples: int = 256):
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, n_samples)            # shape (N,)  BUG: should be (N, 1)
    noise = 0.2 * torch.randn(n_samples)
    y = 2 * x + 3 + noise                           # y ~ 2x + 3
    # BUG: returns float64, and x still 1D
    return x, y


def train(num_epochs: int = 10, batch_size: int = 32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    x, y = make_data()
    ds = TensorDataset(x, y)                        # BUG: x,y shapes not 2D
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)  # BUG: no shuffle

    model = TinyRegressor()                         # BUG: not moved to device
    crit = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)  # BUG: very high LR

    for epoch in range(num_epochs):
        running_loss = 0.0

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            # BUG: missing opt.zero_grad()

            preds = model(xb)                       # will error if model still on CPU
            loss = crit(preds, yb)

            loss.backward()
            opt.step()

            running_loss += loss.item() * xb.size(0)

        print(f"[BUGGED] Epoch {epoch+1}: loss={running_loss/len(ds):.4f}")


if __name__ == "__main__":
    train()