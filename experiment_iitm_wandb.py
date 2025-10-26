#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS6886W Assignment 1: VGG6 Hyperparameter Sweep
-----------------------------------------------
This script is for running a series of experiments on the VGG6 model
with the CIFAR-10 dataset. It's designed to sweep through various
hyperparameters like optimizers, learning rates, and activation functions.

All metrics and results are logged using Weights & Biases for visualization
and analysis, which is a requirement for the assignment.

Example usage:
  python experiment_iitm1.py --epochs 50 --wandb-project cs6886w-a1
"""

import os
import time
import csv
import argparse
import random
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import wandb

# --- Utility Functions ---

def seed_everything(seed_value: int = 42):
    """Sets the seed for reproducibility of experiments."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size).item())
        return results

class StatTracker:
    """A simple class to track running averages of metrics."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def record(self, value, num_items=1):
        self.total += value * num_items
        self.count += num_items

    def average(self):
        return self.total / max(1, self.count)

# --- Data Preparation ---

class CutoutAugmentation(object):
    """
    Implements the Cutout data augmentation technique.
    Randomly masks out one or more square patches from an image.
    """
    def __init__(self, num_patches: int, patch_length: int):
        self.num_patches = num_patches
        self.patch_length = patch_length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.num_patches):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.patch_length // 2, 0, h)
            y2 = np.clip(y + self.patch_length // 2, 0, h)
            x1 = np.clip(x - self.patch_length // 2, 0, w)
            x2 = np.clip(x + self.patch_length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
            
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask

def prepare_dataloaders(data_path: str, batch_size: int, num_workers: int = 2, val_size: int = 5000):
    """Prepares the CIFAR-10 train, validation, and test dataloaders."""
    # Normalization stats for CIFAR-10
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    # Augmentations for the training set
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
        CutoutAugmentation(num_patches=1, patch_length=16),
    ])

    # Only normalization for the test set
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    full_train_dataset = datasets.CIFAR10(data_path, train=True, transform=train_transforms, download=True)
    test_dataset = datasets.CIFAR10(data_path, train=False, transform=test_transforms, download=True)

    # Splitting training data into train and validation sets
    train_size = len(full_train_dataset) - val_size
    train_subset, val_subset = random_split(
        full_train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(2025)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, validation_loader, test_loader

# --- Model Definition (VGG6) ---

ACTIVATION_FN_MAP = {
    'relu': nn.ReLU(inplace=True),
    'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'silu': nn.SiLU(inplace=True),
    'gelu': nn.GELU(),
}

class VGG(nn.Module):
    def __init__(self, feature_extractor: nn.Sequential, num_classes: int = 10, last_channel_size: int = 128, dropout_rate: float = 0.0):
        super().__init__()
        self.features = feature_extractor
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(last_channel_size, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def _make_vgg_layers(config: List, use_bn: bool = True, activation: str = 'relu'):
    layers = []
    in_channels = 3
    act_fn = ACTIVATION_FN_MAP.get(activation.lower(), nn.ReLU(inplace=True))
    
    for layer_cfg in config:
        if layer_cfg == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, layer_cfg, kernel_size=3, padding=1, bias=not use_bn)
            if use_bn:
                layers.extend([conv2d, nn.BatchNorm2d(layer_cfg), act_fn])
            else:
                layers.extend([conv2d, act_fn])
            in_channels = layer_cfg
            
    return nn.Sequential(*layers), in_channels

def vgg6(num_classes: int = 10, use_bn: bool = True, activation: str = 'relu', dropout_rate: float = 0.0):
    vgg6_config = [64, 64, 'M', 128, 128, 'M']
    features, last_channels = _make_vgg_layers(vgg6_config, use_bn=use_bn, activation=activation)
    return VGG(features, num_classes=num_classes, last_channel_size=last_channels, dropout_rate=dropout_rate)

# --- Training and Evaluation Logic ---

def get_optimizer(model_params, name: str, lr: float, weight_decay: float, momentum: float = 0.9):
    """A factory function to build the optimizer based on a name string."""
    optimizer_map = {
        'sgd': optim.SGD,
        'nesterov': lambda p, **k: optim.SGD(p, nesterov=True, **k),
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'adagrad': optim.Adagrad,
        'rmsprop': optim.RMSprop,
        'nadam': optim.NAdam,
    }
    name = name.lower()
    if name not in optimizer_map:
        raise ValueError(f"Unsupported optimizer: {name}")

    # Prepare kwargs, some optimizers don't use momentum
    kwargs = {'lr': lr, 'weight_decay': weight_decay}
    if name in ['sgd', 'nesterov', 'rmsprop']:
        kwargs['momentum'] = momentum
    
    return optimizer_map[name](model_params, **kwargs)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch_num):
    model.train()
    loss_tracker, acc_tracker = StatTracker(), StatTracker()

    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        acc1, = calculate_accuracy(outputs, targets, topk=(1,))
        loss_tracker.record(loss.item(), num_items=inputs.size(0))
        acc_tracker.record(acc1, num_items=inputs.size(0))

        # Log to W&B every 50 batches
        if i % 50 == 0:
            step = epoch_num * len(dataloader) + i
            wandb.log({'batch_train_loss': loss.item(), 'batch_train_acc': acc1}, step=step)

    return loss_tracker.average(), acc_tracker.average()

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    loss_tracker, acc_tracker = StatTracker(), StatTracker()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc1, = calculate_accuracy(outputs, targets, topk=(1,))
            loss_tracker.record(loss.item(), num_items=inputs.size(0))
            acc_tracker.record(acc1, num_items=inputs.size(0))

    return loss_tracker.average(), acc_tracker.average()

# --- Experiment Setup ---

def setup_hyperparameter_configs() -> List[Dict]:
    """Generates a list of hyperparameter configurations for the sweep."""
    activations = ['relu', 'gelu', 'silu']
    optimizers = ['sgd', 'nesterov', 'adam', 'adamw']
    lr_schedules = {'sgd': [0.1, 0.01], 'nesterov': [0.1, 0.01], 'adam': [0.001, 0.0005], 'adamw': [0.001, 0.0005]}
    batch_sizes = [64, 128]
    scheduler_types = {'sgd': 'multistep', 'nesterov': 'multistep', 'adam': 'cosine', 'adamw': 'cosine'}
    
    configs = []
    config_id = 1
    for act in activations:
        for opt in optimizers:
            for lr in lr_schedules[opt]:
                for bs in batch_sizes:
                    configs.append({
                        'id': f'cfg_{config_id:02d}',
                        'activation': act,
                        'optimizer': opt,
                        'lr': lr,
                        'batch_size': bs,
                        'use_bn': True,
                        'dropout_rate': 0.3 if bs == 64 else 0.0, # More regularization for smaller batches
                        'weight_decay': 5e-4,
                        'momentum': 0.9,
                        'scheduler': scheduler_types[opt],
                    })
                    config_id += 1
    return configs[:24] # Capping at 24 runs

# --- Main Execution ---

def execute_training_run(config: Dict, args) -> Dict:
    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    run_name = f"{config['id']}_{config['activation']}_{config['optimizer']}_lr{config['lr']}_bs{config['batch_size']}"
    
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=config, # Log the entire config dict
        reinit=True,
    )
    
    train_loader, val_loader, test_loader = prepare_dataloaders(
        data_path=args.data_root,
        batch_size=config['batch_size'],
        num_workers=args.num_workers,
        val_size=args.val_split,
    )

    model = vgg6(
        use_bn=config['use_bn'],
        activation=config['activation'],
        dropout_rate=config['dropout_rate'],
    ).to(device)
    
    wandb.watch(model, criterion=nn.CrossEntropyLoss(), log='all', log_freq=100)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model.parameters(), name=config['optimizer'], lr=config['lr'], weight_decay=config['weight_decay'], momentum=config['momentum'])
    
    scheduler = None
    if config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif config['scheduler'] == 'multistep':
        milestones = [int(0.6 * args.epochs), int(0.8 * args.epochs)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        
    start_time = time.time()
    best_val_accuracy = -1.0
    best_epoch = -1
    best_model_state = None

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch + 1
            # Save model state to CPU to avoid using up GPU memory
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if scheduler:
            scheduler.step()
        
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }, step=epoch)

        print(f"[{config['id']}] Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # Load best model and evaluate on the test set
    if best_model_state:
        model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    
    total_time = time.time() - start_time
    
    # Log final summary metrics to W&B
    wandb.summary['best_val_acc'] = best_val_accuracy
    wandb.summary['test_acc_at_best_val'] = test_acc
    wandb.summary['best_epoch'] = best_epoch
    wandb.summary['total_runtime_sec'] = total_time
    
    if args.save_checkpoints:
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt_path = os.path.join(args.save_dir, f"{run_name}.pt")
        torch.save({'model_state': best_model_state, 'config': config}, ckpt_path)
        wandb.save(ckpt_path)
        
    result_summary = {
        'id': config['id'],
        'activation': config['activation'],
        'optimizer': config['optimizer'],
        'lr': config['lr'],
        'batch_size': config['batch_size'],
        'best_val_acc': round(best_val_accuracy, 2),
        'best_epoch': best_epoch,
        'test_acc_at_best': round(test_acc, 2),
        'runtime_sec': round(total_time, 2),
    }
    
    wandb.finish()
    return result_summary


def main():
    parser = argparse.ArgumentParser(description="VGG6 CIFAR-10 Hyperparameter Sweep")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--data-root', type=str, default='./data', help='Path for dataset storage.')
    parser.add_argument('--csv-path', type=str, default='results.csv', help='Path to save summary CSV.')
    parser.add_argument('--num-workers', type=int, default=2, help='Dataloader workers.')
    parser.add_argument('--val-split', type=int, default=5000, help='Size of validation set.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--limit', type=int, default=0, help='Limit to the first N configs (0 for all).')
    parser.add_argument('--device', type=str, default=None, help='Device to use (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--save-checkpoints', action='store_true', help='Save best model checkpoints.')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Directory for checkpoints.')
    parser.add_argument('--wandb-project', type=str, default='cs6886w-vgg6-sweep', help='W&B project name.')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Your W&B username or team name.')
    args = parser.parse_args()

    all_configs = setup_hyperparameter_configs()
    configs_to_run = all_configs[:args.limit] if args.limit > 0 else all_configs

    print(f"Starting sweep of {len(configs_to_run)} configurations...")
    
    csv_header = ['id','activation','optimizer','lr','batch_size','best_val_acc','best_epoch','test_acc_at_best','runtime_sec']
    file_exists = os.path.exists(args.csv_path)
    with open(args.csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        if not file_exists:
            writer.writeheader()
        
        for config in configs_to_run:
            print("\n" + "="*50)
            print(f"RUNNING CONFIG: {config['id']} | {config['activation']} | {config['optimizer']} | lr={config['lr']} | bs={config['batch_size']}")
            print("="*50)
            
            result = execute_training_run(config, args)
            writer.writerow(result)
            f.flush() # Write to file immediately
            
            print(f"Finished {config['id']}. Best Val Acc: {result['best_val_acc']}%, Test Acc: {result['test_acc_at_best']}%")

    print(f"\nSweep complete. All results saved to {os.path.abspath(args.csv_path)}")

if __name__ == '__main__':
    main()