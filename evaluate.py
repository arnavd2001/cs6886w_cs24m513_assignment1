# evaluate.py

"""
Script to evaluate a trained model checkpoint on the CIFAR-10 test set.

This allows for easy verification of a model's performance without
retraining. It loads the model architecture and weights from a specified
checkpoint file.

Usage:
  python evaluate.py --checkpoint-path saved_models/best_model.pt
"""

import argparse
import torch

# Import modular components from the 'src' directory
from src.model import vgg6
from src.data import prepare_dataloaders
from src.utils import StatTracker, calculate_accuracy

def evaluate_checkpoint(checkpoint_path: str, data_path: str):
    """Loads a model and evaluates its performance on the test dataset."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the checkpoint. It's good practice to map to CPU first
    # in case the model was saved on a different device.
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    model_state = checkpoint['model_state']

    # 1. Re-create the model architecture based on the saved config
    print("Instantiating model with the following configuration:")
    print(config)
    model = vgg6(
        use_bn=config.get('use_bn', True),
        activation=config['activation'],
        dropout_rate=config['dropout_rate']
    ).to(device)

    # 2. Load the trained weights into the model
    model.load_state_dict(model_state)
    model.eval()

    # 3. Prepare the test dataloader
    # We only need the test loader, so we can ignore train and val
    _, _, test_loader = prepare_dataloaders(
        data_path=data_path,
        batch_size=config['batch_size'],
        num_workers=2,
        val_size=5000  # This value doesn't affect the test set
    )

    # 4. Perform evaluation
    criterion = torch.nn.CrossEntropyLoss()
    loss_tracker = StatTracker()
    acc_tracker = StatTracker()

    print("\nStarting evaluation on the test set...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            acc1, = calculate_accuracy(outputs, targets, topk=(1,))
            
            loss_tracker.record(loss.item(), num_items=inputs.size(0))
            acc_tracker.record(acc1, num_items=inputs.size(0))

    # [cite_start]5. Report the final metrics [cite: 21]
    final_loss = loss_tracker.average()
    final_acc = acc_tracker.average()
    
    print("\n" + "="*30)
    print("      Evaluation Complete      ")
    print("="*30)
    print(f"  Test Loss: {final_loss:.4f}")
    print(f"  Test Top-1 Accuracy: {final_acc:.2f}%")
    print("="*30)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a VGG6 checkpoint.")
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to the model checkpoint file (.pt).')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Path to the CIFAR-10 dataset.')
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint_path, args.data_root)

if __name__ == '__main__':
    main()

