# exercise7_solution.py
# CLEAN SOLUTION FOR EXERCISE 7
# --------------------------------------------------------------

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def make_data(n=500):
    torch.manual_seed(0)

    X = torch.randn(n, 50)

    y_raw = X[:, 3] * 0.8 + X[:, 7] * -1.2 + X[:, 12] * 0.5

    # Correct: encode labels as {0,1}
    y = (y_raw > 0).float()

    return X, y


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 32)
        self.fc2 = nn.Linear(32, 1)  # one logit

    def forward(self, x):
        logits = self.fc2(torch.relu(self.fc1(x)))
        return logits  # shape (batch,1)


def train(epochs=5, batch_size=32, lr=1e-2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    X, y = make_data()

    # Correct: normalize across features (dim=0)
    X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-6)

    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = Net().to(device)

    # Correct loss for binary classification w/ raw logits
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()

            logits = model(xb).squeeze(1)  # shape (batch,)

            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)

            # predictions: sigmoid(logits) > 0.5
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            correct += (preds == yb).sum().item()
            total += xb.size(0)

        print(f"[SOLVED] Epoch {epoch+1} loss={total_loss/total:.4f}, acc={correct/total:.3f}")


if __name__ == "__main__":
    train()