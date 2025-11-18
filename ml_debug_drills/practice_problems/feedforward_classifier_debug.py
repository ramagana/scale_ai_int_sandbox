"""
feedforward_classifier_debug.py

INTENTIONALLY BUGGY / INCOMPLETE

Mini-exercise:
- Synthetic 2D classification problem.
- Feedforward network.
- Several issues are present (dtype, shapes, grad handling, etc.).
- Run this file, observe errors or bad behavior, and fix step-by-step.

Suggestions:
- Use VS Code / Codespaces debugger.
- Inspect tensor shapes, dtypes, and .grad.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_synthetic_data(n_samples: int = 512):
    """
    Create a toy binary classification dataset in 2D.
    Class 0: centered near (-1, -1)
    Class 1: centered near (+1, +1)
    """
    torch.manual_seed(0)

    n_half = n_samples // 2

    # Class 0
    x0 = torch.randn(n_half, 2) * 0.5 + torch.tensor([-1.0, -1.0])
    y0 = torch.zeros(n_half)  # BUG 1: dtype will later be wrong for CrossEntropyLoss

    # Class 1
    x1 = torch.randn(n_half, 2) * 0.5 + torch.tensor([1.0, 1.0])
    y1 = torch.ones(n_half)

    X = torch.cat([x0, x1], dim=0)
    y = torch.cat([y0, y1], dim=0)

    # BUG 1 detail: y is float by default; CrossEntropyLoss expects Long indices
    # TODO: Fix label dtype when you debug.
    return X, y


class FeedforwardNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, num_classes=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # BUG 2: wrong output dimension for 2-class classification (should be num_classes)
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    X, y = make_synthetic_data(n_samples=512)

    # BUG 3 (soft): not putting data on the same device as the model later
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = FeedforwardNet(input_dim=2, hidden_dim=32, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    num_epochs = 5

    # BUG 4 (critical): no_grad used around training loop - kills gradient flow
    # This will cause loss.backward() to fail or do nothing useful.
    with torch.no_grad():
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for xb, yb in loader:
                # BUG 3: Model is on `device`, inputs are still on CPU
                # TODO: Move tensors to device.
                xb = xb
                yb = yb  # TODO: and fix dtype for labels

                # BUG 5 (soft): missing optimizer.zero_grad() leads to grad accumulation
                # optimizer.zero_grad()

                logits = model(xb)
                # BUG 2 shows up here: logits.shape is (batch, 1), but we want (batch, num_classes)
                loss = criterion(logits, yb)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * xb.size(0)

                # Compute accuracy
                # BUG 6: prediction logic might be wrong given logits shape
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

            epoch_loss = running_loss / len(dataset)
            acc = correct / total if total > 0 else 0.0
            print(f"[DEBUG] Epoch {epoch+1}: loss={epoch_loss:.4f}, acc={acc:.3f}")


if __name__ == "__main__":
    main()