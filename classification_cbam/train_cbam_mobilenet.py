# train_cbam_mobilenet.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import BLBDataset
from cbam import CBAM
import os

class CBAMMobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cbam_img = CBAM(3)
        self.cbam_mask = CBAM(3)
        self.fuse_conv = nn.Conv2d(6, 3, 1)
        self.mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.mobilenet.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, image, mask):
        img_enh = self.cbam_img(image)
        mask_enh = self.cbam_mask(mask)
        fused = torch.cat([img_enh, mask_enh], dim=1)
        fused = self.fuse_conv(fused)
        return self.mobilenet(fused)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5
num_epochs = 20
batch_size = 16
learning_rate = 1e-4

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = BLBDataset("data/train", "data/train", transform)
val_dataset = BLBDataset("data/val", "data/val", transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = CBAMMobileNetV2(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for imgs, masks, labels in train_loader:
        imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for imgs, masks, labels in val_loader:
            imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
            outputs = model(imgs, masks)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader.dataset)
    val_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] TrainLoss: {epoch_loss:.4f} ValLoss: {val_loss:.4f} ValAcc: {val_acc:.4f}")
