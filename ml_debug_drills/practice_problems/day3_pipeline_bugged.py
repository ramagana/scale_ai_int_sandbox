# day3_pipeline_bugged.py
#
# INTENTIONALLY BUGGY VERSION of the tabular pipeline.
# Compare against day3_pipeline_solution.py after you debug.

import csv
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class TabularDataset(Dataset):
    def __init__(self, path: str):
        self.features = []
        self.labels = []

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # BUG 1: no handling for missing fields; also no explicit error or logging
                f1 = row["f1"]             # still strings here
                f2 = row["f2"]
                y = row["y"]

                # BUG 2: values not converted to float/int before storing
                # they stay as strings, which will confuse tensor creation
                self.features.append([f1, f2])
                self.labels.append(y)

        # BUG 3: tensors built without explicit dtype
        # This may produce unexpected dtypes (e.g., string-like or default float),
        # and labels will NOT be Long.
        self.X = torch.tensor(self.features)   # wrong
        self.y = torch.tensor(self.labels)     # wrong

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # BUG 4 (soft): returns as-is; downstream dtype/shape errors will appear in training loop
        return self.X[idx], self.y[idx]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            # BUG 5: wrong output size for 2-class CE (uses 1 output instead of 2)
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def train_loop(csv_path: str, num_epochs: int = 5, batch_size: int = 32):
    ds = TabularDataset(csv_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Net().to(device)

    # BUG 6: CrossEntropyLoss is fine, but labels/dtypes coming in will be wrong.
    crit = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    # BUG 7: sanity check does not verify dtype or shape, just prints
    xb0, yb0 = next(iter(dl))
    print("xb0:", xb0.shape, xb0.dtype, xb0.device)
    print("yb0:", yb0.shape, yb0.dtype, yb0.device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in dl:
            # BUG 8: not moving data to device (model is on GPU potentially)
            # xb = xb.to(device)
            # yb = yb.to(device)

            # BUG 9: zero_grad is after backward (wrong placement)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.zero_grad()
            opt.step()

            running_loss += loss.item() * xb.size(0)

            # BUG 10: logits shape is (batch, 1); argmax over dim=1 gives all zeros,
            # and yb dtype/device may not match.
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        print(f"[BUGGED] Epoch {epoch+1}: loss={running_loss/len(ds):.4f}, acc={correct/total:.3f}")


if __name__ == "__main__":
    # reuse write_dummy_csv from earlier or create your own CSV first
    train_loop("toy_data.csv")