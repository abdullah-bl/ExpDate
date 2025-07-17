#!/usr/bin/env python3
"""
Test script for MPS functionality and basic pipeline operation
"""

import torch
import time
import sys
from PIL import Image
import numpy as np

def test_mps_availability():
    """Test MPS availability and basic functionality"""
    print("=" * 60)
    print("MPS AVAILABILITY TEST")
    print("=" * 60)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check MPS availability
    mps_available = torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")
    
    if mps_available:
        print("✅ MPS is available!")
    else:
        print("❌ MPS is not available")
        print("   This could be due to:")
        print("   - macOS version < 12.3")
        print("   - PyTorch version < 2.0")
        print("   - Non-Apple Silicon Mac")
        return False
    
    # Test basic tensor operations
    try:
        device = torch.device("mps")
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = torch.matmul(x, y)
        print("✅ Basic tensor operations work on MPS")
    except Exception as e:
        print(f"❌ Basic tensor operations failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test model loading and basic forward pass"""
    print("\n" + "=" * 60)
    print("MODEL LOADING TEST")
    print("=" * 60)
    
    try:
        # Import models
        from train.date_detector import DateDetector
        from train.dmy_detector import DMYDetector
        from train.dan_recognizer import DAN
        from utils.mps_utils import DEVICE, DTYPE
        
        print(f"Current device: {DEVICE}")
        print(f"Current dtype: {DTYPE}")
        
        # Test DateDetector
        print("\nTesting DateDetector...")
        date_model = DateDetector()
        dummy_input = torch.randn(1, 3, 800, 1333).to(DEVICE)
        if DEVICE.type == "mps":
            dummy_input = dummy_input.to(dtype=DTYPE)
        
        with torch.no_grad():
            cls_logits, bbox_reg, centerness = date_model(dummy_input)
        print("✅ DateDetector forward pass successful")
        
        # Test DMYDetector
        print("\nTesting DMYDetector...")
        dmy_model = DMYDetector()
        dummy_input = torch.randn(1, 3, 64, 256).to(DEVICE)
        if DEVICE.type == "mps":
            dummy_input = dummy_input.to(dtype=DTYPE)
        
        with torch.no_grad():
            cls_logits, bbox_reg, centerness = dmy_model(dummy_input)
        print("✅ DMYDetector forward pass successful")
        
        # Test DAN
        print("\nTesting DAN...")
        dan_model = DAN()
        dummy_input = torch.randn(1, 3, 32, 128).to(DEVICE)
        if DEVICE.type == "mps":
            dummy_input = dummy_input.to(dtype=DTYPE)
        
        with torch.no_grad():
            logits = dan_model(dummy_input)
        print("✅ DAN forward pass successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline():
    """Test the complete pipeline"""
    print("\n" + "=" * 60)
    print("PIPELINE TEST")
    print("=" * 60)
    
    try:
        from pipeline import ExpDatePipeline
        
        # Create a dummy image
        dummy_image = Image.new('RGB', (800, 1333), color='white')
        dummy_image.save("test_image.jpg")
        
        # Initialize pipeline
        print("Initializing pipeline...")
        pipeline = ExpDatePipeline("dummy_date.pth", "dummy_dmy.pth", "dummy_rec.pth")
        
        # Test inference
        print("Running inference...")
        start_time = time.time()
        result = pipeline("test_image.jpg")
        inference_time = time.time() - start_time
        
        print(f"✅ Pipeline inference successful")
        print(f"   Result: {result}")
        print(f"   Time: {inference_time:.3f}s")
        
        # Clean up
        import os
        if os.path.exists("test_image.jpg"):
            os.remove("test_image.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_management():
    """Test memory management utilities"""
    print("\n" + "=" * 60)
    print("MEMORY MANAGEMENT TEST")
    print("=" * 60)
    
    try:
        from utils.mps_utils import get_memory_info, clear_memory
        
        # Get initial memory info
        initial_memory = get_memory_info()
        print(f"Initial memory: {initial_memory['allocated']:.2f}GB allocated")
        
        # Create some tensors to use memory
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        tensors = []
        for i in range(10):
            tensor = torch.randn(1000, 1000).to(device)
            tensors.append(tensor)
        
        # Check memory after allocation
        after_allocation = get_memory_info()
        print(f"After allocation: {after_allocation['allocated']:.2f}GB allocated")
        
        # Clear memory
        clear_memory()
        
        # Check memory after clearing
        after_clear = get_memory_info()
        print(f"After clearing: {after_clear['allocated']:.2f}GB allocated")
        
        print("✅ Memory management utilities work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        return False

def test_data_loading():
    """Test data loading utilities"""
    print("\n" + "=" * 60)
    print("DATA LOADING TEST")
    print("=" * 60)
    
    try:
        from train.data import create_dummy_dataloader
        
        # Create dummy dataloader
        dataloader = create_dummy_dataloader(batch_size=2, num_samples=4)
        
        # Test iteration
        for i, (imgs, targets) in enumerate(dataloader):
            print(f"Batch {i+1}:")
            print(f"  Images shape: {imgs.shape}")
            print(f"  Targets keys: {list(targets.keys())}")
            if i >= 1:  # Just test first 2 batches
                break
        
        print("✅ Data loading utilities work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("MPS-OPTIMIZED EXPIRATION DATE RECOGNITION - TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("MPS Availability", test_mps_availability),
        ("Model Loading", test_model_loading),
        ("Pipeline", test_pipeline),
        ("Memory Management", test_memory_management),
        ("Data Loading", test_data_loading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your MPS setup is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 