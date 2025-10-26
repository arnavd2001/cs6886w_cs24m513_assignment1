# run_sweep.py

"""
Main script for running the hyperparameter sweep for CS6886W Assignment 1.

This script generates a predefined set of hyperparameter configurations,
then iterates through each one, training the VGG6 model from scratch.
All metrics are logged to Weights & Biases for analysis and plotting.

Usage:
  python run_sweep.py --epochs 50 --wandb-project cs6886w-a1 --limit 5
"""

import os
import csv
import argparse
from typing import List, Dict

# Import modular components from the 'src' directory
from src.trainer import execute_training_run
from src.utils import seed_everything

def setup_hyperparameter_configs() -> List[Dict]:
    """
    Generates a deterministic list of hyperparameter configurations for the sweep.
    This covers variations in activation functions, optimizers, learning rates,
    [cite_start]and batch sizes as required by the assignment. [cite: 23, 25, 27]
    """
    activations = ['relu', 'gelu', 'silu', 'tanh']
    optimizers = ['sgd', 'nesterov', 'adam', 'adamw', 'rmsprop']
    lr_schedules = {
        'sgd': [0.1, 0.01], 'nesterov': [0.1, 0.01],
        'adam': [0.001, 0.0005], 'adamw': [0.001, 0.0005],
        'rmsprop': [0.001, 0.0001]
    }
    batch_sizes = [64, 128]
    scheduler_types = {
        'sgd': 'multistep', 'nesterov': 'multistep',
        'adam': 'cosine', 'adamw': 'cosine', 'rmsprop': 'cosine'
    }

    configs = []
    config_id = 1
    for act in activations:
        for opt in optimizers:
            if opt not in lr_schedules: continue
            for lr in lr_schedules[opt]:
                for bs in batch_sizes:
                    configs.append({
                        'id': f'cfg_{config_id:02d}',
                        'activation': act,
                        'optimizer': opt,
                        'lr': lr,
                        'batch_size': bs,
                        'use_bn': True,
                        'dropout_rate': 0.3 if bs == 64 else 0.0,
                        'weight_decay': 5e-4,
                        'momentum': 0.9,
                        'scheduler': scheduler_types[opt],
                    })
                    config_id += 1
    return configs

def main():
    """Parses arguments and orchestrates the hyperparameter sweep."""
    parser = argparse.ArgumentParser(description="VGG6 CIFAR-10 Hyperparameter Sweep")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--data-root', type=str, default='./data', help='Path for dataset storage.')
    parser.add_argument('--csv-path', type=str, default='results/sweep_results.csv', help='Path to save summary CSV.')
    parser.add_argument('--num-workers', type=int, default=2, help='Dataloader workers.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--limit', type=int, default=0, help='Limit to the first N configs (0 for all).')
    parser.add_argument('--wandb-project', type=str, required=True, help='W&B project name.')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Your W&B username or team name.')
    args = parser.parse_args()

    # Ensure reproducibility
    seed_everything(args.seed)

    # Prepare directories
    os.makedirs('results', exist_ok=True)

    all_configs = setup_hyperparameter_configs()
    configs_to_run = all_configs[:args.limit] if args.limit > 0 else all_configs

    print(f"Starting sweep of {len(configs_to_run)} configurations...")
    print(f"Results will be logged to W&B project: '{args.wandb_project}'")

    # Prepare CSV file for logging results locally
    csv_header = ['id','activation','optimizer','lr','batch_size','best_val_acc','best_epoch','test_acc_at_best','runtime_sec']
    file_exists = os.path.exists(args.csv_path)
    with open(args.csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        if not file_exists:
            writer.writeheader()

        for config in configs_to_run:
            print("\n" + "="*60)
            print(f"RUNNING CONFIG: {config['id']} | {config['activation']} | {config['optimizer']} | lr={config['lr']} | bs={config['batch_size']}")
            print("="*60)

            # Each configuration is a full training run managed by the trainer module
            result = execute_training_run(config, args)

            writer.writerow(result)
            f.flush()  # Write to file after each run to prevent data loss

            print(f"Finished {config['id']}. Best Val Acc: {result['best_val_acc']}%, Test Acc: {result['test_acc_at_best']}%")

    print(f"\nSweep complete. All results saved to {os.path.abspath(args.csv_path)}")

if __name__ == '__main__':
    main()
