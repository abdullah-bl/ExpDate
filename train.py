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
from train.dan_train import train_dan_recognizer


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
        train_date_detector(config, output_dir=args.output_dir)

    if args.component in ['all', 'dmy']:
        print("\n=== Training DMY Detector ===")
        train_dmy_detector(config, output_dir=args.output_dir)

    if args.component in ['all', 'dan']:
        print("\n=== Training DAN Recognizer ===")
        train_dan_recognizer(config, output_dir=args.output_dir)

    print("\n=== Training Complete ===")
    print(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 