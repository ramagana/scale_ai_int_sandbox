"""
day3_tabular_debug.py

INTENTIONALLY BUGGY: read tabular data, convert to tensors, feed model.
You’ll debug:
- bad dtypes
- shape mismatch
- missing device move
"""

import csv
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import logging

CSV_PATH = "toy_data.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def safe_load_dataset(path: str):
    try:
        ds = TabularDataset(path)
        logging.info("Loaded dataset with %d rows", len(ds))
        return ds
    except FileNotFoundError as e:
        logging.error("File not found: %s", path)
        raise
    except Exception as e:
        logging.exception("Failed to load dataset")
        raise



def write_dummy_csv(path: str = CSV_PATH, n_rows: int = 20):
    # Write a simple CSV: features f1, f2, target y ∈ {0,1}
    import random

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["f1", "f2", "y"])
        rng = random.Random(0)
        for _ in range(n_rows):
            f1 = rng.uniform(-1, 1)
            f2 = rng.uniform(-1, 1)
            y = 1 if f1 + f2 > 0 else 0
            writer.writerow([f1, f2, y])


class TabularDataset(Dataset):
    def __init__(self, path: str):
        self.features = []
        self.labels = []

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # BUG 1: strings kept as-is → not converted to float
                if row["f1"] == '' or row["f2"] == '' or row["y"] == '':
                    raise ValueError(f"Missing data in row: {row}")     # skip rows with missing data
                f1 = float(row["f1"])  
                f2 = float(row["f2"])
                y = int(row["y"])
   
                self.features.append([f1, f2])
                self.labels.append(y)

        # BUG 2: tensors built without dtype, so everything becomes string/float weirdness
        self.X = torch.tensor(self.features, dtype=torch.float32)
        self.y = torch.tensor(self.labels, dtype=torch.long)   # wrong

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    write_dummy_csv(CSV_PATH)
    # update 2:  reorder,  device to top .  and move all inputs,  optimizer, models to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ds = safe_load_dataset(CSV_PATH)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    
    # One-time validation check
    xb0, yb0 = next(iter(loader))
    logging.info(
        "Sample batch: xb.shape=%s xb.dtype=%s xb.device=%s | yb.shape=%s yb.dtype=%s yb.device=%s",
        xb0.shape, xb0.dtype, xb0.device,
        yb0.shape, yb0.dtype, yb0.device,
    )

    assert xb0.ndim == 2, f"Expected 2D features, got shape {xb0.shape}"
    assert yb0.ndim == 1, f"Expected 1D labels, got shape {yb0.shape}"
    assert xb0.dtype == torch.float32, f"Expected float32 features, got {xb0.dtype}"
    assert yb0.dtype == torch.long, f"Expected long labels, got {yb0.dtype}"        
    
    model = SmallNet().to(device)
    crit = nn.CrossEntropyLoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    

    for epoch in range(3):
        running_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # BUG 3: device mismatch (model on device, xb/yb on CPU)
            # BUG 4: CrossEntropyLoss expects Long labels and float inputs
            optim.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optim.step()
            running_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}, loss={running_loss/len(ds):.4f}")


if __name__ == "__main__":
    main()