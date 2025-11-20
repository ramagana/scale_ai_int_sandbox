# exercise9_working.py
#
# Task:
#   Train a small CNN on CIFAR-10 (10 classes, 3x32x32 images).

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-2


def get_dataloaders(batch_size=BATCH_SIZE):
    transform_train = transforms.Compose([
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
        transforms.ToTensor(),
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


class BadCifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 16 * 16, NUM_CLASSES)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probs = self.softmax(logits)
        return probs


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_loader, test_loader = get_dataloaders()

    model = BadCifarNet().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        # TRAIN
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            opt.zero_grad()
            outputs = model(xb)
            loss = crit(outputs, yb)

            loss.backward()
            opt.step()

            train_loss += loss.item() * xb.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)

        # VAL
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        for xb, yb in test_loader:
            outputs = model(xb)
            loss = crit(outputs, yb)

            val_loss += loss.item() * xb.size(0)
            preds = outputs.argmax(dim=1)
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