#!/usr/bin/env python3
"""
Train U-Net Segmentation Model for Lensed Source Detection

Trains a U-Net model to detect lensed source pixels in images.
Uses inverse engineering: train on simulation data where we know the ground truth.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from prism.ml.segmentation_model import train_segmentation_model
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not available")


def main():
    parser = argparse.ArgumentParser(
        description="Train U-Net segmentation model for lensed source detection"
    )
    parser.add_argument(
        '--training-dir',
        type=str,
        required=True,
        help='Directory with training data (images/, masks/, training_metadata.csv)'
    )
    parser.add_argument(
        '--model-output',
        type=str,
        default=None,
        help='Path to save trained model (default: training_dir/model.pth)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (cpu or cuda)'
    )
    
    args = parser.parse_args()
    
    if not TORCH_AVAILABLE:
        print("❌ Error: PyTorch not available. Install with: pip install torch")
        sys.exit(1)
    
    training_dir = Path(args.training_dir)
    if not training_dir.exists():
        print(f"❌ Error: Training directory not found: {training_dir}")
        sys.exit(1)
    
    if args.model_output:
        model_path = Path(args.model_output)
    else:
        model_path = training_dir / 'unet_segmentation_model.pth'
    
    print(f"🎯 Training U-Net segmentation model...")
    print(f"   Training data: {training_dir}")
    print(f"   Model output: {model_path}")
    
    train_segmentation_model(
        training_dir,
        model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )


if __name__ == '__main__':
    main()

