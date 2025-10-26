import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# --- Set random seed for reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- Data transforms and loader ---
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length
    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

def get_cifar10_loaders(data_root, batch_size, num_workers=2, val_split=5000):
    train_tfms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout(n_holes=1, length=16),
    ]
    test_tfms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    full_train = datasets.CIFAR10(data_root, train=True, transform=transforms.Compose(train_tfms), download=True)
    test_data = datasets.CIFAR10(data_root, train=False, transform=transforms.Compose(test_tfms), download=True)
    if val_split > 0 and val_split < len(full_train):
        train_len = len(full_train) - val_split
        train_data, val_data = random_split(full_train, [train_len, val_split], generator=torch.Generator().manual_seed(2025))
    else:
        train_data, val_data = full_train, None
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if val_data is not None else None
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# --- Model definition ---
ACT_MAP = {
    'relu': nn.ReLU(inplace=True),
    'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'silu': nn.SiLU(inplace=True),
    'gelu': nn.GELU(),
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=10, last_c=128, dropout=0.0):
        super().__init__()
        self.features = features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(last_c, num_classes),
        )
        self._init_weights()
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

def make_layers(cfg, in_channels=3, bn=True, activation='relu', width_mult=1.0):
    layers = []
    act = ACT_MAP.get(activation.lower(), nn.ReLU(inplace=True))
    last_channels = None
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            out_c = int(v * width_mult)
            conv2d = nn.Conv2d(in_channels, out_c, kernel_size=3, padding=1, bias=not bn)
            if bn:
                layers += [conv2d, nn.BatchNorm2d(out_c), act]
            else:
                layers += [conv2d, act]
            in_channels = out_c
            last_channels = out_c
    return nn.Sequential(*layers), last_channels or int(128 * width_mult)

def vgg6(num_classes=10, bn=True, activation='relu', width_mult=1.0, dropout=0.0):
    cfg_vgg6 = [64, 64, 'M', 128, 128, 'M']
    features, last_c = make_layers(cfg_vgg6, bn=bn, activation=activation, width_mult=width_mult)
    return VGG(features, num_classes=num_classes, last_c=last_c, dropout=dropout)

# --- Training and evaluation ---
def accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        return 100.0 * correct / target.size(0)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc, total_samples = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        acc = accuracy(out, y)
        total_loss += loss.item() * x.size(0)
        total_acc += acc * x.size(0)
        total_samples += x.size(0)
    return total_loss / total_samples, total_acc / total_samples

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, total_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            acc = accuracy(out, y)
            total_loss += loss.item() * x.size(0)
            total_acc += acc * x.size(0)
            total_samples += x.size(0)
    return total_loss / total_samples, total_acc / total_samples

# --- Best configuration (C22) ---
activation = 'gelu'
optimizer_name = 'nesterov'
lr = 0.1
batch_size = 128
bn = True
dropout = 0.0
scheduler_name = 'multistep'
weight_decay = 0.0005
momentum = 0.9
width_mult = 1.0
epochs = 20
seed = 42
val_split = 5000
device = torch.device('cpu')

# --- Prepare data ---
train_loader, val_loader, test_loader = get_cifar10_loaders('./data', batch_size, num_workers=2, val_split=val_split)

# --- Build model ---
model = vgg6(num_classes=10, bn=bn, activation=activation, width_mult=width_mult, dropout=dropout).to(device)
criterion = nn.CrossEntropyLoss()
if optimizer_name == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
elif optimizer_name == 'nesterov':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
elif optimizer_name == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
elif optimizer_name == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
else:
    raise ValueError("Unsupported optimizer")

if scheduler_name == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
elif scheduler_name == 'multistep':
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*epochs), int(0.8*epochs)], gamma=0.1)
else:
    scheduler = None

# --- Training loop ---
best_val_acc = -1.0
best_epoch = -1
best_state = None

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if scheduler is not None:
        scheduler.step()
    print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}")

# --- Evaluate best model on test set ---
if best_state is not None:
    model.load_state_dict(best_state)
test_loss, test_acc = evaluate(model, test_loader, criterion, device)

print("\n=== Final Results for Best Config (C22) ===")
print(f"Best Validation Accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
print(f"Test Accuracy at Best: {test_acc:.2f}%")

