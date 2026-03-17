"""Quick test to verify inference on 1 sample"""
import os
os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from inference_group_classification import (
    InferenceConfig,
    load_data,
    get_split_from_info,
    GroupClassificationInference
)

# Load config
config = InferenceConfig()

# Load data
print("Loading data...")
full_data = load_data(config)

# Get validation split
_, val_data, groups = get_split_from_info(config, full_data)

print(f"\nLoaded {len(val_data)} validation samples")
print(f"Groups: {groups}")

# Take just first sample
test_sample = val_data[0]

print("\n" + "="*70)
print("TEST SAMPLE:")
print("="*70)
print(f"ID: {test_sample['id']}")
print(f"File: {test_sample['file_name']}")
print(f"Disease: {test_sample['disease']}")
print(f"Ground Truth Group: {test_sample.get('disease_group', 'NOT SET')}")
print("="*70)

# Load model
print("\nLoading model...")
model = GroupClassificationInference(config.MODEL_PATH_MERGED)

# Run inference on just this one sample
print("\nRunning inference on 1 sample...")
result = model.predict_single(test_sample['image_path'])

print("\n" + "="*70)
print("RESULT:")
print("="*70)
print(f"Ground Truth: {test_sample.get('disease_group', 'NOT SET')}")
print(f"Predicted: {result.get('predicted_group', 'NOT SET')}")
print(f"Status: {result.get('status', 'NOT SET')}")
print("="*70)
