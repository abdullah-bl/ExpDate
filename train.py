#!/usr/bin/env python3
"""
Main training script for the ExpDate model pipeline.
This script orchestrates the training of all components in the model.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from config.config import ExpDateConfig
from train.date_detector_train import train_date_detector
from train.dmy_detector_train import train_dmy_detector
from train.dan_train import train_dan


def main():
    parser = argparse.ArgumentParser(description='Train the ExpDate model pipeline')
    parser.add_argument('--config', type=str, default='config/config.json',
                        help='Path to configuration file')
    parser.add_argument('--component', type=str, choices=['all', 'date', 'dmy', 'dan'], default='all',
                        help='Which component to train')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for trained models')
    args = parser.parse_args()

    # Load configuration
    config = ExpDateConfig(args.config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Training component: {args.component}")
    print(f"Output directory: {args.output_dir}")

    if args.component in ['all', 'date']:
        print("\n=== Training Date Detector ===")
        from train.date_detector import DateDetector
        from train.data import build_dataloader, create_dummy_dataloader
        
        # Initialize model
        model = DateDetector()
        
        # Setup data loader
        try:
            dataloader = build_dataloader()
        except Exception as e:
            print(f"Warning: Could not create data loader: {e}")
            print("Creating dummy data loader for demonstration...")
            dataloader = create_dummy_dataloader(batch_size=4)
        
        # Train with output path
        save_path = os.path.join(args.output_dir, "date_detector_mps.pth")
        train_date_detector(model, dataloader, save_path=save_path)

    if args.component in ['all', 'dmy']:
        print("\n=== Training DMY Detector ===")
        from train.dmy_detector import DMYDetector
        from train.data import build_dataloader, create_dummy_dataloader
        
        # Initialize model
        model = DMYDetector()
        
        # Setup data loader
        try:
            dataloader = build_dataloader()
        except Exception as e:
            print(f"Warning: Could not create data loader: {e}")
            print("Creating dummy data loader for demonstration...")
            dataloader = create_dummy_dataloader(batch_size=4)
        
        # Train with output path
        save_path = os.path.join(args.output_dir, "dmy_detector_mps.pth")
        train_dmy_detector(model, dataloader, save_path=save_path)

    if args.component in ['all', 'dan']:
        print("\n=== Training DAN Recognizer ===")
        from train.dan_recognizer import DAN
        from train.data import build_dataloader, create_dummy_dataloader
        
        # Initialize model
        model = DAN()
        
        # Setup data loader
        try:
            dataloader = build_dataloader()
        except Exception as e:
            print(f"Warning: Could not create data loader: {e}")
            print("Creating dummy data loader for demonstration...")
            dataloader = create_dummy_dataloader(batch_size=8)
        
        # Train with output path
        save_path = os.path.join(args.output_dir, "dan_recognizer_mps.pth")
        train_dan(model, dataloader, save_path=save_path)

    print("\n=== Training Complete ===")
    print(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 