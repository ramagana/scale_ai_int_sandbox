"""
exercise4_solution.py
Correct 4-class MLP classifier.
"""

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


NUM_CLASSES = 4
FEATURES = 20


def make_data(n_samples=400):
    torch.manual_seed(0)
    X = torch.randn(n_samples, FEATURES) * 10 + 50

    # class based on 4 feature-group sums
    y = torch.argmin(
        torch.stack([
            X[:, :5].sum(dim=1),
            X[:, 5:10].sum(dim=1),
            X[:, 10:15].sum(dim=1),
            X[:, 15:].sum(dim=1),
        ], dim=1),
        dim=1,
    )

    # return integer labels (not one-hot)
    return X.float(), y.long()


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),  # correct
        )

    def forward(self, x):
        return self.net(x)  # no softmax – CE expects logits


def train(num_epochs=5, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    X, y = make_data()
    ds = TensorDataset(X, y)

    # proper split
    n_train = int(len(ds) * 0.8)
    ds_train, ds_val = torch.utils.data.random_split(ds, [n_train, len(ds) - n_train])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size)

    model = Classifier().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total = 0

        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total += xb.size(0)

        print(f"Epoch {epoch+1}: loss={total_loss/total:.4f}, acc={total_correct/total:.3f}")

    print("Training complete.")


if __name__ == "__main__":
    train()