# CS6886W Assignment 1: VGG6 Hyperparameter Exploration on CIFAR-10

This repository contains the implementation for the CS6886W (System Engineering for Deep Learning) Assignment 1. The primary objective is to train a VGG6 neural network on the CIFAR-10 dataset and systematically analyze its performance across a wide range of hyperparameter configurations.

The core script, `experiment_iitm1.py`, automates the process of training the model with over 20 predefined configurations, logging the results, and saving the best-performing model checkpoints.

## Key Features

- VGG6 Architecture: A lightweight VGG-style model implemented in PyTorch, configurable with different activation functions, batch normalization, and dropout layers.
- CIFAR-10 Data Pipeline: Efficient data loading for CIFAR-10 using `torchvision`, with standard augmentations like `RandomCrop`, `RandomHorizontalFlip`, and `Cutout` for regularization.
- Automated Experiment Runner: Systematically sweeps through a predefined set of hyperparameters, including activation functions, optimizers, learning rates, and batch sizes.
- Comprehensive Logging: All results, including validation accuracy, test accuracy, and runtime, are logged to a structured CSV file for easy analysis and plotting.
- Reproducibility: The entire pipeline is seeded to ensure that experimental results are reproducible.
- Checkpointing: The script saves the model state corresponding to the epoch with the highest validation accuracy for each run.

## Repository Structure

```
.
├── checkpoints/
├── data/
├── experiment_iitm1.py
├── results.csv
├── requirements.txt
└── README.md
```

## Setup and Installation

Follow these steps to set up the environment and run the experiments.

1.  Clone the Repository
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  Create a Virtual Environment (Recommended)
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install Dependencies
    Install all the required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main script `experiment_iitm1.py` is controlled via command-line arguments.

### Running the Experiments

To run all 24 predefined configurations for 50 epochs and save the model checkpoints:
```bash
python experiment_iitm1.py --epochs 50 --save-checkpoints
```

To run a limited subset of configurations (e.g., the first 5) for a quick test:
```bash
python experiment_iitm1.py --epochs 10 --limit 5
```

### Command-Line Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--epochs` | `int` | `20` | Number of epochs to train each configuration. |
| `--data-root` | `str` | `./data` | Path to the CIFAR-10 dataset directory. |
| `--csv-path` | `str` | `results.csv` | Path to the output CSV file for storing results. |
| `--num-workers` | `int` | `2` | Number of worker processes for the data loader. |
| `--val-split` | `int` | `5000` | Number of training samples to use for the validation set. |
| `--seed` | `int` | `42` | Random seed for reproducibility. |
| `--limit` | `int` | `0` | Limit the run to the first N configurations (0 means all). |
| `--device` | `str` | `None` | Manually specify device (e.g., `cuda`, `cpu`). Auto-detects if `None`. |
| `--save-checkpoints` | `store_true` | `False` | If set, saves the best model checkpoint for each run. |
| `--save-dir` | `str` | `./checkpoints` | Directory to save model checkpoints. |

## Experiment Configuration Space

The script deterministically generates 24 configurations to explore the hyperparameter space. The parameters being varied are:

- Activation Functions: `relu`, `gelu`, `silu`
- Optimizers: `sgd`, `nesterov`, `adam`, `adamw`
- Learning Rates: `[0.1, 0.01]` for SGD-based optimizers and `[0.001, 0.0005]` for Adam-based optimizers
- Batch Sizes: `64`, `128`
- Schedulers: `MultiStepLR` for SGD-based optimizers and `CosineAnnealingLR` for Adam-based optimizers
- Dropout: `0.3` for batch size 64, `0.0` for batch size 128.

All runs use a fixed `weight_decay` of `5e-4` and a `momentum` of `0.9` (where applicable).

## Results

The performance of each configuration is logged to the console during execution and saved permanently in the CSV file specified by `--csv-path`.

The output CSV file (`results.csv`) will contain the following columns, allowing for detailed analysis and plotting:
`id`, `activation`, `optimizer`, `lr`, `batch_size`, `bn`, `dropout`, `scheduler`, `weight_decay`, `momentum`, `width_mult`, `epochs`, `seed`, `val_split`, `best_val_acc`, `best_epoch`, `test_acc_at_best`, `runtime_sec`, `device`.