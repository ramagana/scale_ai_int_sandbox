# exercise7_working.py
# Same as exercise7_bugged.py but without inline BUG annotations.

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def make_data(n=500):
    torch.manual_seed(0)

    X = torch.randn(n, 50)

    y_raw = X[:, 3] * 0.8 + X[:, 7] * -1.2 + X[:, 12] * 0.5 # float 

    y = torch.where(y_raw > 0, -1.0, 1.0)

    return X, y


class BadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


def train(epochs=5, batch_size=32, lr=1e-2, n_sample=500):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    X, y = make_data(n_sample)
    y =(y==1).long()


    X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-6) # bug

    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True) # bug: shuffle on
    # todo: add validation check/print
    print(f"validation of first minibatch")
    xb0, yb0 =  next(iter(dl))
    print(f"xb0 : \t{xb0.shape}\t{xb0.dtype}\t{xb0.device}")
    print(f" yb0: \t{yb0.shape}\t{yb0.dtype}\t{yb0.device}")

    # 2. model/opt/loss
    model = BadNet().to(device)

    #crit = nn.BCELoss() # todo: check if loss has sigmoid
    # changing the loss function to cross entropy loss to match the target data
    crit = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Train loop 
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_correct = 0
        total = 0

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()

            logits = model(xb)
            # todo: check logits and if they sum 
            loss = crit(logits, yb)

            loss.backward()
            opt.step()

            epoch_loss += loss.item() * xb.size(0)

            # preds = (logits > 0.5).float() # change to argmax
            preds = logits.argmax(dim=1)

            correct = (preds == yb).sum().item()
            epoch_correct += correct
            total += xb.size(0)

        print(f"[WORKING] Epoch {epoch+1} loss={epoch_loss/total:.4f}, acc={epoch_correct/total:.3f}")


if __name__ == "__main__":
    train(epochs=10, n_sample=500, batch_size=16)