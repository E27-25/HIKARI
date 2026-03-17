"""
CUDA Diagnostic Script
Run this first to verify your GPU setup before training.
"""

import sys

print("=" * 50)
print("CUDA DIAGNOSTIC CHECK")
print("=" * 50)

# Check Python version
print(f"\n[1] Python Version: {sys.version}")

# Check PyTorch and CUDA
print("\n[2] PyTorch & CUDA Check:")
try:
    import torch
    print(f"    PyTorch Version: {torch.__version__}")
    print(f"    CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"    CUDA Version: {torch.version.cuda}")
        print(f"    cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"    GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n    GPU {i}: {props.name}")
            print(f"        Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"        Compute Capability: {props.major}.{props.minor}")
            print(f"        Multi Processors: {props.multi_processor_count}")
        
        # Test CUDA with simple operation
        print("\n[3] CUDA Test:")
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        y = x * 2
        print(f"    Simple CUDA operation: PASSED")
        print(f"    Result: {y.cpu().numpy()}")
        
        # Check memory
        print("\n[4] GPU Memory:")
        print(f"    Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"    Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
        # BF16 support
        print("\n[5] BF16 Support:")
        print(f"    BF16 Supported: {torch.cuda.is_bf16_supported()}")
        
    else:
        print("\n    ❌ CUDA NOT AVAILABLE!")
        print("\n    Possible fixes:")
        print("    1. Install NVIDIA drivers: https://www.nvidia.com/drivers")
        print("    2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        print("    3. Reinstall PyTorch with CUDA:")
        print("       pip uninstall torch")
        print("       pip install torch --index-url https://download.pytorch.org/whl/cu121")
        
except ImportError as e:
    print(f"    ❌ PyTorch not installed: {e}")
    print("    Install: pip install torch --index-url https://download.pytorch.org/whl/cu121")

# Check bitsandbytes
print("\n[6] Bitsandbytes Check:")
try:
    import bitsandbytes as bnb
    print(f"    Version: {bnb.__version__}")
    print("    Status: OK")
except ImportError:
    print("    ❌ Not installed: pip install bitsandbytes")
except Exception as e:
    print(f"    ❌ Error: {e}")

# Check unsloth
print("\n[7] Unsloth Check:")
try:
    from unsloth import FastVisionModel
    print("    FastVisionModel: OK")
except ImportError as e:
    print(f"    ❌ Not installed or error: {e}")

print("\n" + "=" * 50)
print("DIAGNOSTIC COMPLETE")
print("=" * 50)
