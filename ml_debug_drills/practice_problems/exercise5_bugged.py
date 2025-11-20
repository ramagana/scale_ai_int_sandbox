"""
exercise5_bugged.py
INTENTIONALLY BUGGY

Task:
    Train a 2-class classifier on tabular data.
    Includes:
      - normalization step
      - shallow MLP
      - MSE loss incorrectly used
      - several subtle shape/device bugs
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_data(n=300):
    torch.manual_seed(0)
    X = torch.randn(n, 6) * 20 + 100   # big numeric range
    y = (X[:, 0] + X[:, 3] > 100).float()  # 0/1 labels (float)

    return X, y


class WeirdNet(nn.Module):
    def __init__(self):
        super().__init__()
        # BUG 1: No hidden layer bias. Weird but allowed.
        # BUG 2: Output dimension = 2 but MSE loss used later!
        self.fc1 = nn.Linear(6, 16, bias=False)
        self.fc2 = nn.Linear(16, 2)  # logits

    def forward(self, x):
        z = self.fc1(x)
        z = torch.relu(z) * 1000  # BUG 3: exploding activation scale
        return self.fc2(z)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ----- Data -----
    X, y = make_data()

    # BUG 4: wrong normalization (dim=0 instead of dim=1)
    X_norm = (X - X.mean(dim=1, keepdim=True)) / (X.std(dim=1, keepdim=True) + 1e-6)

    # BUG 5: labels should be long for CE, not float
    y = y  # unchanged (float)

    ds = TensorDataset(X_norm, y)
    dl = DataLoader(ds, batch_size=64, shuffle=False)  # BUG 6: shuffle=False

    # ----- Model / loss / opt -----
    model = WeirdNet().to(device)
    crit = nn.MSELoss()  # BUG 7: wrong loss for classification
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(3):
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in dl:
            # BUG 8: forgetting to move labels to device
            xb = xb.to(device)

            # BUG 9: missing zero_grad()
            logits = model(xb)

            # BUG 10: shape mismatch (logits are (batch,2), y is (batch,))
            loss = crit(logits, yb)  

            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)

            # BUG 11: accuracy: comparing 2D logits to float labels
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        print(f"[BUGGED] Epoch {epoch+1}: loss={total_loss/len(ds):.4f}, acc={correct/total:.3f}")


if __name__ == "__main__":
    train()