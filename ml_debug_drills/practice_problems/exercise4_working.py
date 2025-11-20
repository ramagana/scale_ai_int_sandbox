"""
exercise4_bugged.py
INTENTIONALLY BUGGY

Task:
    Train a 4-class classifier on synthetic tabular data with 20 features.

Bugs to find:

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


    y_oh = torch.nn.functional.one_hot(y, num_classes=NUM_CLASSES).float()

    return X.float(), y_oh  # wrong labels output


class BadClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 4),   
            # nn.Softmax(dim=1), 
        )

    def forward(self, x):
        return self.net(x)


def train(num_epochs=5, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    X, y = make_data()
    y = y.argmax(dim=1)
    
    ds = TensorDataset(X, y)


    dl_train = DataLoader(ds, batch_size=batch_size, shuffle=False)
    dl_val = DataLoader(ds, batch_size=batch_size, shuffle=False)

    print('validation check prior to training and val loops')
    xt0, yt0 = next(iter(dl_train))
    xv0, yv0 = next(iter(dl_val))
    print(f"train X:\t{xt0.shape} shape, \t{xt0.dtype} dtype, \t{xt0.device}")
    print(f"train y:\t{yt0.shape} shape, \t{yt0.dtype} dtype, \t{yt0.device}")

    print(f"val X:\t{xv0.shape} shape, \t{xv0.dtype} dtype, \t{xv0.device}")
    print(f"val y:\t{yv0.shape} shape, \t{yv0.dtype} dtype, \t{yv0.device}")

    model = BadClassifier().to(device) 
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"model on device: {next(model.parameters()).device}")
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total = 0

        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb) 
            loss = crit(logits, yb) 

            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)

            # preds = (logits > 0.5).float()
            preds = logits.argmax(dim=1) 
            correct = (preds == yb).sum().item()
            total_correct += correct
            total += xb.size(0)

        print(f"[BUGGED] Epoch {epoch+1}: loss={total_loss/len(ds):.4f}, acc={total_correct/total:.3f}")


if __name__ == "__main__":
    # train(num_epochs=1, batch_size=400)
    train(num_epochs=15,  batch_size=40)