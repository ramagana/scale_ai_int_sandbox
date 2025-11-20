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
        self.fc1 = nn.Linear(6, 16, bias=False)
        self.fc2 = nn.Linear(16, 2)  # logits

    def forward(self, x):
        z = self.fc1(x)
        z = torch.relu(z)# * 1000 
        return self.fc2(z)


def train(epoch_num=3, batch_size=32, lr=0.1, n_samples=300):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ----- Data -----
    X, y = make_data(n=n_samples)

    X_norm = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-6)

    y = y.long()

    ds = TensorDataset(X_norm, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)  # BUG: shuffle

     # ----- Validation step of first minibatch-----
    xb0, yb0 = next(iter(dl) )
    
    print(f"Validation step of first minibatch")
    print(f"xb0:\t{xb0.shape}\t{xb0.dtype}\t{xb0.device}")
    print(f"yb0:\t{yb0.shape}\t{yb0.dtype}\t{yb0.device}")
    # ----- Model / loss / opt -----
    model = WeirdNet().to(device)
    crit = nn.CrossEntropyLoss() 
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epoch_num):
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()

            logits = model(xb)
            if epoch == 0:
                print("logits min/max:", logits.min().item(), logits.max().item())

            loss = crit(logits, yb)  

            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        print(f"[BUGGED] Epoch {epoch+1}: loss={total_loss/len(ds):.4f}, acc={correct/total:.3f}")


if __name__ == "__main__":
    train(epoch_num=10, batch_size=64, lr=0.1, n_samples=300)