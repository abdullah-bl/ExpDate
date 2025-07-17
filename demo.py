#!/usr/bin/env python3
"""
Demo script for MPS-optimized Expiration Date Recognition Pipeline

This script demonstrates the complete pipeline for recognizing expiration dates
from images using Apple Silicon's Metal Performance Shaders (MPS).

Usage:
    python demo.py --image path/to/image.jpg
    python demo.py --test_all
"""

import argparse
import os
import time
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from pipeline import ExpDatePipeline
from utils.mps_utils import DEVICE, DTYPE, get_memory_info, clear_memory

def create_dummy_checkpoints():
    """Create dummy checkpoint files for demonstration"""
    print("Creating dummy checkpoint files...")
    
    # Create dummy models and save them
    from train.date_detector import DateDetector
    from train.dmy_detector import DMYDetector
    from train.dan_recognizer import DAN
    
    # Date detector
    date_model = DateDetector()
    torch.save(date_model.state_dict(), "dummy_date.pth")
    
    # DMY detector
    dmy_model = DMYDetector()
    torch.save(dmy_model.state_dict(), "dummy_dmy.pth")
    
    # DAN recognizer
    dan_model = DAN()
    torch.save(dan_model.state_dict(), "dummy_rec.pth")
    
    print("Dummy checkpoints created successfully!")

def test_single_image(image_path: str):
    """Test the pipeline on a single image"""
    print(f"\n{'='*60}")
    print(f"Testing on image: {image_path}")
    print(f"{'='*60}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Create pipeline
    print("Initializing pipeline...")
    pipeline = ExpDatePipeline("dummy_date.pth", "dummy_dmy.pth", "dummy_rec.pth")
    
    # Process image
    print("Processing image...")
    start_time = time.time()
    
    try:
        result = pipeline(image_path)
        processing_time = time.time() - start_time
        
        print(f"\nResults:")
        print(f"  Detected Date: {result}")
        print(f"  Processing Time: {processing_time:.3f}s")
        print(f"  Device: {DEVICE}")
        print(f"  Dtype: {DTYPE}")
        
        # Memory usage
        memory_info = get_memory_info()
        print(f"  Memory Usage: {memory_info['allocated']:.2f}GB allocated")
        
    except Exception as e:
        print(f"Error processing image: {e}")
    
    # Clear memory
    clear_memory()

def test_all_images():
    """Test the pipeline on all available test images"""
    print(f"\n{'='*60}")
    print("Testing on all available images")
    print(f"{'='*60}")
    
    # Find test images
    test_dir = "data/evaluation/images"
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return
    
    image_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    if not image_files:
        print("No test images found")
        return
    
    print(f"Found {len(image_files)} test images")
    
    # Create pipeline
    print("Initializing pipeline...")
    pipeline = ExpDatePipeline("dummy_date.pth", "dummy_dmy.pth", "dummy_rec.pth")
    
    # Process each image
    results = []
    total_time = 0
    
    for i, img_file in enumerate(image_files[:10]):  # Limit to first 10 for demo
        img_path = os.path.join(test_dir, img_file)
        print(f"\nProcessing {i+1}/{min(10, len(image_files))}: {img_file}")
        
        start_time = time.time()
        try:
            result = pipeline(img_path)
            processing_time = time.time() - start_time
            total_time += processing_time
            
            results.append({
                'file': img_file,
                'result': result,
                'time': processing_time
            })
            
            print(f"  Result: {result}")
            print(f"  Time: {processing_time:.3f}s")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'file': img_file,
                'result': 'ERROR',
                'time': 0
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {len(results)}")
    print(f"Average processing time: {total_time/len(results):.3f}s")
    print(f"Total time: {total_time:.3f}s")
    
    # Memory usage
    memory_info = get_memory_info()
    print(f"Final memory usage: {memory_info['allocated']:.2f}GB allocated")
    
    # Clear memory
    clear_memory()

def benchmark_performance():
    """Benchmark the performance on different devices"""
    print(f"\n{'='*60}")
    print("Performance Benchmark")
    print(f"{'='*60}")
    
    # Create a dummy image for benchmarking
    dummy_image = Image.new('RGB', (800, 1333), color='white')
    dummy_image.save("dummy_test.jpg")
    
    # Test on different devices
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    if torch.cuda.is_available():
        devices.append('cuda')
    
    results = {}
    
    for device in devices:
        print(f"\nTesting on {device.upper()}...")
        
        # Set device
        torch_device = torch.device(device)
        
        # Create pipeline with device-specific checkpoints
        pipeline = ExpDatePipeline("dummy_date.pth", "dummy_dmy.pth", "dummy_rec.pth")
        
        # Warm up
        for _ in range(3):
            try:
                pipeline("dummy_test.jpg")
            except:
                pass
        
        # Benchmark
        times = []
        for _ in range(10):
            start_time = time.time()
            try:
                pipeline("dummy_test.jpg")
                times.append(time.time() - start_time)
            except:
                pass
        
        if times:
            avg_time = sum(times) / len(times)
            results[device] = avg_time
            print(f"  Average time: {avg_time:.3f}s")
        else:
            print(f"  Failed to run on {device}")
    
    # Clean up
    if os.path.exists("dummy_test.jpg"):
        os.remove("dummy_test.jpg")
    
    # Print comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    for device, time_taken in results.items():
        print(f"{device.upper()}: {time_taken:.3f}s")
    
    # Clear memory
    clear_memory()

def show_system_info():
    """Display system information"""
    print(f"\n{'='*60}")
    print("SYSTEM INFORMATION")
    print(f"{'='*60}")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Current device: {DEVICE}")
    print(f"Current dtype: {DTYPE}")
    
    # Device capabilities
    if torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders): Available")
    else:
        print("MPS (Metal Performance Shaders): Not available")
    
    if torch.cuda.is_available():
        print(f"CUDA: Available ({torch.cuda.get_device_name()})")
    else:
        print("CUDA: Not available")
    
    # Memory information
    memory_info = get_memory_info()
    print(f"Current memory usage: {memory_info['allocated']:.2f}GB allocated")
    
    # Model information
    from train.date_detector import DateDetector
    from train.dmy_detector import DMYDetector
    from train.dan_recognizer import DAN
    
    date_model = DateDetector()
    dmy_model = DMYDetector()
    dan_model = DAN()
    
    total_params = (
        sum(p.numel() for p in date_model.parameters()) +
        sum(p.numel() for p in dmy_model.parameters()) +
        sum(p.numel() for p in dan_model.parameters())
    )
    
    print(f"Total model parameters: {total_params:,}")

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="MPS-optimized Expiration Date Recognition Demo")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--test_all", action="store_true", help="Test on all available images")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--info", action="store_true", help="Show system information")
    
    args = parser.parse_args()
    
    print("MPS-Optimized Expiration Date Recognition Pipeline")
    print("=" * 60)
    
    # Show system info
    show_system_info()
    
    # Create dummy checkpoints if they don't exist
    if not all(os.path.exists(f) for f in ["dummy_date.pth", "dummy_dmy.pth", "dummy_rec.pth"]):
        create_dummy_checkpoints()
    
    # Run requested tests
    if args.info:
        return  # Already shown above
    
    if args.benchmark:
        benchmark_performance()
    
    if args.test_all:
        test_all_images()
    
    if args.image:
        test_single_image(args.image)
    
    # Default: test on a sample image if available
    if not any([args.image, args.test_all, args.benchmark, args.info]):
        sample_image = "data/evaluation/images/test_00001.jpg"
        if os.path.exists(sample_image):
            test_single_image(sample_image)
        else:
            print("\nNo specific test requested. Use --help for options.")
            print("Available options:")
            print("  --image <path>     Test on specific image")
            print("  --test_all         Test on all available images")
            print("  --benchmark        Run performance benchmark")
            print("  --info             Show system information")

if __name__ == "__main__":
    main() 