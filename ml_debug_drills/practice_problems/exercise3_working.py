# exercise3_bugged.py
#
# INTENTIONALLY BUGGED:
# Tiny MLP to fit y = 2x + 3 + noise (1-D regression).
#
# Bugs to find & fix:
# 

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
    x = torch.linspace(-1, 1, n_samples)          
    noise = 0.2 * torch.randn(n_samples)
    y = 2 * x + 3 + noise                           
    
    return x, y


def train(num_epochs: int = 10, batch_size: int = 32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    x, y = make_data()
    x = x.reshape(-1,1).float()
    y = y.reshape(-1,1).float()
    ds = TensorDataset(x, y)                        
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = TinyRegressor().to(device)                       
    crit = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.05) 

    xb0, ybo = next(iter(dl))
    print(f"Sanity batch xb0: shape={xb0.shape}, dtype={xb0.dtype}, device={xb0.device}")
    print(f"Sanity batch yb0: shape={ybo.shape}, dtype={ybo.dtype}, device={ybo.device}")

    for epoch in range(num_epochs):
        running_loss = 0.0

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            preds = model(xb)                      
            loss = crit(preds, yb)

            loss.backward()
            opt.step()

            running_loss += loss.item() * xb.size(0)

        print(f"[BUGGED] Epoch {epoch+1}: loss={running_loss/len(ds):.4f}")


if __name__ == "__main__":
    train(num_epochs=10, batch_size=32) 
