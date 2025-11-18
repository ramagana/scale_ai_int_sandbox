# train_mlp_bugged.py
"""
INTENTIONALLY BUGGY version of the tiny MLP script.
Your job: run this, fix the errors, and make it behave like train_mlp_solution.py.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset  # BUG 1: forgot to import TensorDataset


def make_data(n_samples: int = 512):
    torch.manual_seed(0)
    x = torch.randn(n_samples, 2)
    # BUG 2: labels are bool, not long ints; CrossEntropyLoss expects class indices (LongTensor)
    y = (x[:, 0] + x[:, 1] > 0)
    return x, y


class SimpleMLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=16, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

def convert_xb_yb(xb, yb):
    if yb.dtype != torch.long:
        yb = yb.long()
    if xb != torch.float64:
        xb.float()  # BUG 2 fix: convert bool to long
    return xb, yb

def train(num_epochs: int = 5, batch_size: int = 32, lr: float = 1e-2):
    x, y = make_data()
    # BUG 1 shows up here: TensorDataset is not imported
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for xb, yb in loader: # specify training_loader
            # BUG 3 (soft): forgot optimizer.zero_grad(), gradients will accumulate
            xb, yb = convert_xb_yb(xb, yb)  # BUG 2: forgot to convert bool to long)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print("Epoch", epoch + 1, "loss=", epoch_loss)

    with torch.no_grad():
        xb, yb = next(iter(loader)) # todo: change to import validation_loader
        xb, yb = convert_xb_yb(xb, yb)             
        preds = model(xb).argmax(dim=1)
        acc = (preds == yb).float().mean().item()
        print(f"Sample batch accuracy: {acc:.3f}")


if __name__ == "__main__":
    train()