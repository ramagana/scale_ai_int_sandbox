# exercise9_solution.py
#
# Clean CIFAR-10 CNN training script.
# Fixes:
#   - Correct transform order: ToTensor() then Normalize
#   - BatchNorm2d used for conv feature maps
#   - Correct flatten size for Linear layer
#   - No softmax before CrossEntropyLoss (use raw logits)
#   - Proper device usage
#   - model.train() / model.eval() + torch.no_grad() for val

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3  # safer than 1e-2 but you could keep 1e-2 too


def get_dataloaders(batch_size=BATCH_SIZE):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
        transforms.RandomHorizontalFlip(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train,
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # (B,16,32,32)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,16,16,16)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (B,32,16,16)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,32,8,8)
        )
        self.classifier = nn.Linear(32 * 8 * 8, NUM_CLASSES)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten to (B, 32*8*8)
        logits = self.classifier(x)
        return logits  # raw scores for CrossEntropyLoss


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_loader, test_loader = get_dataloaders()

    model = CifarNet().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        # TRAIN
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            train_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)

        # VAL
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = crit(logits, yb)

                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        print(
            f"Epoch {epoch+1} "
            f"train_loss={train_loss/train_total:.4f} "
            f"train_acc={train_correct/train_total:.3f} | "
            f"val_loss={val_loss/val_total:.4f} "
            f"val_acc={val_correct/val_total:.3f}"
        )


if __name__ == "__main__":
    train()