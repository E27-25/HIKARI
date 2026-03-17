#!/usr/bin/env python3
"""
Quick test script for dataset formatting parallelization.
Tests different num_proc values to find optimal setting for your system.
"""

import os
import time
import pandas as pd
from PIL import Image
from datasets import Dataset
import psutil
import platform

# Disable torch.compile
os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

print("=" * 70)
print("DATASET FORMATTING PARALLEL PROCESSING TEST")
print("=" * 70)
print(f"Platform: {platform.system()} {platform.release()}")
print(f"CPU cores: {psutil.cpu_count()}")
print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
print("=" * 70)


# Configuration
CSV_PATH = "./SkinCAP/skincap_v240623.csv"
IMAGE_BASE_PATH = "./SkinCAP/skincap"
TEST_SIZE = 200  # Small subset for quick testing


def convert_classification(sample):
    """Simplified conversion function (same as training)"""
    image = Image.open(sample["image_path"]).convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["prompt"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["answer"]},
            ],
        },
    ]
    return {"messages": conversation}


def test_num_proc(dataset, num_proc, test_name):
    """Test dataset formatting with specific num_proc value"""
    print(f"\n{'=' * 70}")
    print(f"TEST: {test_name}")
    print(f"num_proc = {num_proc}")
    print("=" * 70)

    try:
        start_time = time.time()

        formatted_dataset = dataset.map(
            convert_classification,
            remove_columns=["image_path", "prompt", "answer"],
            desc=f"Formatting with num_proc={num_proc}",
            num_proc=num_proc,
        )

        elapsed = time.time() - start_time
        samples_per_sec = len(dataset) / elapsed

        print(f"✅ SUCCESS!")
        print(f"   Time: {elapsed:.2f} seconds")
        print(f"   Speed: {samples_per_sec:.2f} samples/sec")

        return {
            "test": test_name,
            "num_proc": num_proc,
            "success": True,
            "time": elapsed,
            "speed": samples_per_sec,
            "error": None
        }

    except Exception as e:
        print(f"❌ FAILED!")
        print(f"   Error: {type(e).__name__}: {str(e)[:100]}")

        return {
            "test": test_name,
            "num_proc": num_proc,
            "success": False,
            "time": None,
            "speed": None,
            "error": str(e)[:200]
        }


def main():
    print("\n[1/4] Loading dataset...")

    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: CSV file not found at {CSV_PATH}")
        print("Please update CSV_PATH in this script to point to your data.")
        return

    # Load data
    df = pd.read_csv(CSV_PATH)
    print(f"   Total samples in CSV: {len(df)}")
    print(f"   Columns found: {', '.join(df.columns[:10])}...")

    # Take small subset for testing
    df_test = df.head(TEST_SIZE).copy()
    print(f"   Using {len(df_test)} samples for testing")

    # Prepare simple test data
    test_data = []
    for idx, row in df_test.iterrows():
        # Try different column names for image path
        if 'skincap_file_path' in row and pd.notna(row['skincap_file_path']):
            image_path = os.path.join(IMAGE_BASE_PATH, row['skincap_file_path'])
        elif 'DDI' in row and pd.notna(row['DDI']):
            image_path = os.path.join(IMAGE_BASE_PATH, row['DDI'])
        elif 'image_path' in row and pd.notna(row['image_path']):
            image_path = row['image_path']
        else:
            continue

        if os.path.exists(image_path):
            # Get disease/answer
            answer = row.get('disease_group', row.get('disease', 'Unknown'))

            test_data.append({
                "image_path": image_path,
                "prompt": "What disease is shown in this image?",
                "answer": str(answer)
            })

    print(f"   Found {len(test_data)} valid images")

    if len(test_data) < 10:
        print("❌ Error: Not enough valid images found. Check IMAGE_BASE_PATH.")
        return

    # Create dataset
    dataset = Dataset.from_list(test_data)
    print(f"   Dataset created: {len(dataset)} samples")

    # Run tests
    print("\n[2/4] Running tests...")
    results = []

    # Test 1: Single-threaded (baseline)
    results.append(test_num_proc(dataset, 1, "Baseline (Single-threaded)"))

    # Test 2: 2 processes (Windows default)
    results.append(test_num_proc(dataset, 2, "Windows Default"))

    # Test 3: 4 processes (aggressive)
    results.append(test_num_proc(dataset, 4, "Aggressive"))

    # Test 4: None (let datasets library decide)
    results.append(test_num_proc(dataset, None, "Auto (None)"))

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Test':<30} {'num_proc':<12} {'Status':<10} {'Time':<12} {'Speedup':<10}")
    print("-" * 70)

    baseline_time = None
    for r in results:
        if r['success']:
            if baseline_time is None:
                baseline_time = r['time']
                speedup = "1.00x"
            else:
                speedup = f"{baseline_time / r['time']:.2f}x"

            print(f"{r['test']:<30} {str(r['num_proc']):<12} {'✅ PASS':<10} "
                  f"{r['time']:>7.2f}s    {speedup:<10}")
        else:
            print(f"{r['test']:<30} {str(r['num_proc']):<12} {'❌ FAIL':<10} "
                  f"{'N/A':>7}     {'N/A':<10}")

    print("=" * 70)

    # Recommendations
    print("\n[3/4] Recommendations:")

    successful_tests = [r for r in results if r['success'] and r['num_proc'] is not None]
    if not successful_tests:
        print("❌ No multiprocessing tests succeeded.")
        print("   Recommendation: Use --dataset_num_proc 1 (single-threaded)")
    else:
        # Find fastest successful test
        fastest = min(successful_tests, key=lambda x: x['time'])
        print(f"✅ Best performing: num_proc={fastest['num_proc']}")
        print(f"   Time: {fastest['time']:.2f}s")
        print(f"   Speedup: {baseline_time / fastest['time']:.2f}x faster than single-threaded")
        print(f"\n   Use in training:")
        print(f"   python train_three_stage_hybrid_topk.py --mode stage1 --dataset_num_proc {fastest['num_proc']}")

    # Error details
    failed_tests = [r for r in results if not r['success']]
    if failed_tests:
        print("\n[4/4] Error Details:")
        for r in failed_tests:
            print(f"\n   {r['test']} (num_proc={r['num_proc']}):")
            print(f"   {r['error']}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
