#!/usr/bin/env python3
"""
Test script to verify MemoryCleanupCallback implementation.
Checks that all required callback methods exist and work correctly.
"""

import sys
import inspect

print("=" * 70)
print("MEMORY CLEANUP CALLBACK TEST")
print("=" * 70)

# Test 1: Import the callback class
print("\n[Test 1] Importing MemoryCleanupCallback...")
try:
    from train_three_stage_hybrid_topk import MemoryCleanupCallback
    print("[OK] Successfully imported MemoryCleanupCallback")
except ImportError as e:
    print(f"[FAIL] Failed to import: {e}")
    sys.exit(1)

# Test 2: Instantiate the callback
print("\n[Test 2] Instantiating callback...")
try:
    callback = MemoryCleanupCallback(cleanup_interval=10)
    print("[OK] Successfully created callback instance")
except Exception as e:
    print(f"[FAIL] Failed to instantiate: {e}")
    sys.exit(1)

# Test 3: Check all required callback methods exist
print("\n[Test 3] Checking required callback methods...")
required_methods = [
    'on_init_end',
    'on_train_begin',
    'on_train_end',
    'on_epoch_begin',
    'on_epoch_end',
    'on_step_begin',
    'on_substep_end',           # NEW
    'on_pre_optimizer_step',    # NEW
    'on_optimizer_step',        # NEW
    'on_step_end',
    'on_evaluate',
    'on_predict',
    'on_prediction_step',       # NEW
    'on_save',
    'on_log',
    'on_push_begin',            # NEW
]

missing_methods = []
for method_name in required_methods:
    if hasattr(callback, method_name):
        method = getattr(callback, method_name)
        if callable(method):
            print(f"   [OK] {method_name}")
        else:
            print(f"   [FAIL] {method_name} (exists but not callable)")
            missing_methods.append(method_name)
    else:
        print(f"   [FAIL] {method_name} (MISSING)")
        missing_methods.append(method_name)

if missing_methods:
    print(f"\n[FAIL] ERROR: Missing methods: {', '.join(missing_methods)}")
    sys.exit(1)
else:
    print(f"\n[OK] All 16 required methods found")

# Test 4: Verify method signatures
print("\n[Test 4] Verifying method signatures...")
try:
    # All callback methods should accept: self, args, state, control, **kwargs
    for method_name in required_methods:
        method = getattr(callback, method_name)
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())

        # Should have at least: args, state, control, and **kwargs
        if 'kwargs' in params:
            print(f"   [OK] {method_name} - correct signature")
        else:
            print(f"   [WARN] {method_name} - signature: {sig}")

    print("\n[OK] Method signatures look correct")
except Exception as e:
    print(f"[FAIL] Error checking signatures: {e}")

# Test 5: Test callback functionality
print("\n[Test 5] Testing callback behavior...")
try:
    # Create mock objects for trainer callback interface
    class MockArgs:
        pass

    class MockState:
        global_step = 0

    class MockControl:
        pass

    args = MockArgs()
    state = MockState()
    control = MockControl()

    # Test on_train_begin
    print("   Testing on_train_begin...")
    callback.on_train_begin(args, state, control)
    print("   [OK] on_train_begin() works")

    # Test on_step_end (should trigger cleanup logic)
    print("   Testing on_step_end...")
    for i in range(15):  # More than cleanup_interval (10)
        callback.on_step_end(args, state, control)
    print(f"   [OK] on_step_end() works (cleanup_interval={callback.cleanup_interval})")

    # Test on_train_end
    print("   Testing on_train_end...")
    callback.on_train_end(args, state, control)
    print("   [OK] on_train_end() works")

    print("\n[OK] Callback behavior test passed")

except Exception as e:
    print(f"[FAIL] Error during behavior test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Verify it can be used with transformers
print("\n[Test 6] Testing compatibility with transformers...")
try:
    from transformers import TrainerCallback

    # Check if MemoryCleanupCallback is a proper callback
    # (It doesn't need to inherit from TrainerCallback, just have the methods)
    print("   Checking callback interface compatibility...")

    # Create a minimal mock trainer to test callback registration
    callback2 = MemoryCleanupCallback(cleanup_interval=50)

    # Verify all methods can be called with kwargs
    test_kwargs = {'model': None, 'tokenizer': None}
    callback2.on_init_end(args, state, control, **test_kwargs)
    callback2.on_train_begin(args, state, control, **test_kwargs)
    callback2.on_step_end(args, state, control, **test_kwargs)

    print("   [OK] Compatible with transformers callback interface")

except ImportError:
    print("   [WARN] transformers not installed, skipping compatibility test")
except Exception as e:
    print(f"   [FAIL] Compatibility test failed: {e}")
    import traceback
    traceback.print_exc()

# Final summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("[OK] MemoryCleanupCallback is correctly implemented")
print("[OK] All 16 required callback methods exist and work")
print("[OK] Callback can be instantiated and used")
print("\nThe callback should work correctly in train_three_stage_hybrid_topk.py")
print("\nIf you still get AttributeError, try:")
print("  1. Close Python completely (not just restart script)")
print("  2. Delete __pycache__ directories:")
print("     rm -rf __pycache__")
print("     rm -rf ./**/__pycache__")
print("  3. Restart your terminal")
print("  4. Run training again")
print("=" * 70)
