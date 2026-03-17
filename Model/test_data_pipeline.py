"""
Data Pipeline Tests for 3-Stage Training Pipeline

Tests all major components of the data preparation pipeline:
1. Data loading from CSV
2. Fuzzy disease label consolidation
3. Group categorization (4-group or 3-group - Strategy G)
4. Top-K filtering (Strategy G — K = all diseases per group)
5. Stratified train/val split
6. Image augmentation

Run before training to catch data pipeline errors early.

Usage:
    python test_data_pipeline.py
"""

from collections import Counter
from pathlib import Path


def test_fuzzy_consolidation():
    """Verify fuzzy matching reduces duplicate labels"""
    print("Running test: Fuzzy consolidation...")
    from train_three_stage_hybrid_topk import fuzzy_consolidate_diseases

    test_data = [
        {"disease": "acne-vulgaris", "image_path": "test1.png", "caption": "test"},
        {"disease": "acne vulgaris", "image_path": "test2.png", "caption": "test"},
        {"disease": "acne_vulgaris", "image_path": "test3.png", "caption": "test"},
        {"disease": "melanoma", "image_path": "test4.png", "caption": "test"},
    ]

    result = fuzzy_consolidate_diseases(test_data, threshold=91)

    unique_diseases = set(item['disease'] for item in result)
    assert len(unique_diseases) == 2, f"Expected 2 unique diseases, got {len(unique_diseases)}: {unique_diseases}"

    # Verify all acne variants are consolidated
    acne_diseases = [item['disease'] for item in result if 'acne' in item['disease']]
    assert len(set(acne_diseases)) == 1, f"Acne variants not consolidated: {set(acne_diseases)}"

    print("  [PASS] Fuzzy consolidation test passed")


def test_categorize_morphology():
    """Verify disease to group mapping (Strategy G - TOP_N aware, GROUP_MODE aware)"""
    print("Running test: Group categorization...")
    from train_three_stage_hybrid_topk import categorize_morphology, Config

    # Diseases present in both top-15 and top-10
    top10_4group_cases = {
        "psoriasis": "1. Inflammatory & Autoimmune Diseases",
        "lupus erythematosus": "1. Inflammatory & Autoimmune Diseases",
        "lichen planus": "1. Inflammatory & Autoimmune Diseases",
        "scleroderma": "1. Inflammatory & Autoimmune Diseases",
        "photodermatoses": "1. Inflammatory & Autoimmune Diseases",
        "sarcoidosis": "1. Inflammatory & Autoimmune Diseases",
        "melanocytic nevi": "2. Benign Tumors, Nevi & Cysts",
        "basal cell carcinoma": "3. Malignant Skin Tumors",
        "squamous cell carcinoma": "3. Malignant Skin Tumors",
        "acne vulgaris": "4. Acne & Follicular Disorders",
    }
    top10_3group_cases = {
        "psoriasis": "1. Inflammatory & Autoimmune Diseases",
        "lupus erythematosus": "1. Inflammatory & Autoimmune Diseases",
        "lichen planus": "1. Inflammatory & Autoimmune Diseases",
        "scleroderma": "1. Inflammatory & Autoimmune Diseases",
        "photodermatoses": "1. Inflammatory & Autoimmune Diseases",
        "sarcoidosis": "1. Inflammatory & Autoimmune Diseases",
        "melanocytic nevi": "2. Benign & Other Non-Malignant",
        "acne vulgaris": "2. Benign & Other Non-Malignant",
        "basal cell carcinoma": "3. Malignant Skin Tumors",
        "squamous cell carcinoma": "3. Malignant Skin Tumors",
    }
    # Extra diseases only in top-15
    top15_extra_4group = {
        "allergic contact dermatitis": "1. Inflammatory & Autoimmune Diseases",
        "neutrophilic dermatoses": "1. Inflammatory & Autoimmune Diseases",
        "seborrheic keratosis": "2. Benign Tumors, Nevi & Cysts",
        "mycosis fungoides": "3. Malignant Skin Tumors",
        "folliculitis": "4. Acne & Follicular Disorders",
    }
    top15_extra_3group = {
        "allergic contact dermatitis": "1. Inflammatory & Autoimmune Diseases",
        "neutrophilic dermatoses": "1. Inflammatory & Autoimmune Diseases",
        "seborrheic keratosis": "2. Benign & Other Non-Malignant",
        "mycosis fungoides": "3. Malignant Skin Tumors",
        "folliculitis": "2. Benign & Other Non-Malignant",
    }

    if Config.TOP_N == 10:
        test_cases = top10_4group_cases if Config.GROUP_MODE == "4group" else top10_3group_cases
        # Diseases removed in top-10 must return "Unknown"
        top10_removed = ["seborrheic keratosis", "allergic contact dermatitis",
                         "neutrophilic dermatoses", "mycosis fungoides", "folliculitis"]
        for disease in top10_removed:
            result = categorize_morphology(disease)
            assert result == "Unknown", (
                f"top-10 mode: '{disease}' should be 'Unknown' but got '{result}'"
            )
        print(f"  [PASS] top-10 removed diseases correctly return 'Unknown'")
    else:  # TOP_N == 15
        if Config.GROUP_MODE == "4group":
            test_cases = {**top10_4group_cases, **top15_extra_4group}
        else:
            test_cases = {**top10_3group_cases, **top15_extra_3group}

    for disease, expected_group in test_cases.items():
        result = categorize_morphology(disease)
        assert result == expected_group, f"{disease} -> '{result}', expected '{expected_group}'"

    # Always test fallback for truly unknown disease
    unknown_result = categorize_morphology("completely_unknown_disease_xyz")
    assert unknown_result == "Unknown", f"Unknown disease -> {unknown_result}, expected 'Unknown'"

    print(f"  [PASS] Group categorization test passed ({len(test_cases)} diseases + fallback, "
          f"mode={Config.GROUP_MODE}, TOP_N={Config.TOP_N})")


def test_topk_filtering():
    """Verify Top-K filtering — Strategy G: K=all diseases so no 'Other' labels"""
    print("Running test: Top-K filtering...")
    from train_three_stage_hybrid_topk import apply_topk_filtering

    # Strategy G: K = number of diseases per group (all diseases named, no "Other")
    # Test with K=2 on a group that has exactly 2 diseases
    test_data = []
    for i in range(60):
        disease = "squamous cell carcinoma" if i < 30 else "basal cell carcinoma"
        test_data.append({
            "disease": disease,
            "disease_group": "3. Malignant Skin Tumors",
            "image_path": f"test{i}.png",
            "caption": "test"
        })

    # K=2 means both diseases kept, no "Other" label
    result, top_diseases = apply_topk_filtering(test_data, {"3. Malignant Skin Tumors": 2})

    labels = set(item['disease_label_stage2'] for item in result)
    assert "squamous cell carcinoma" in labels, "SCC label missing"
    assert "basal cell carcinoma" in labels, "BCC label missing"
    assert "Other_Group3" not in labels, "Unexpected 'Other' label when K=all diseases"

    # Verify top_diseases dict
    assert "3. Malignant Skin Tumors" in top_diseases
    assert len(top_diseases["3. Malignant Skin Tumors"]) == 2

    print("  [PASS] Top-K filtering test passed (no 'Other' label when K=all diseases)")


def test_stratified_split():
    """Verify train/val split maintains proportions"""
    print("Running test: Stratified split...")
    from train_three_stage_hybrid_topk import stratified_split

    # Create test data with known distribution (70% class_A, 30% class_B)
    test_data = []
    for i in range(100):
        test_data.append({
            "disease_label_stage2": "class_A" if i < 70 else "class_B",
            "disease_group": "3. Malignant Skin Tumors",  # Strategy G group name
            "disease": "test_disease",
            "image_path": f"test{i}.png",
            "caption": "test"
        })

    train, val = stratified_split(test_data, test_size=0.1, seed=42)

    assert len(train) == 90, f"Expected 90 train samples, got {len(train)}"
    assert len(val) == 10, f"Expected 10 val samples, got {len(val)}"

    # Check proportions maintained
    train_counts = Counter(item['disease_label_stage2'] for item in train)
    val_counts = Counter(item['disease_label_stage2'] for item in val)

    # 70% of 90 = 63, 70% of 10 = 7
    assert train_counts['class_A'] == 63, f"Train stratification incorrect: got {train_counts['class_A']}, expected 63"
    assert val_counts['class_A'] == 7, f"Val stratification incorrect: got {val_counts['class_A']}, expected 7"

    # 30% of 90 = 27, 30% of 10 = 3
    assert train_counts['class_B'] == 27, f"Train stratification incorrect: got {train_counts['class_B']}, expected 27"
    assert val_counts['class_B'] == 3, f"Val stratification incorrect: got {val_counts['class_B']}, expected 3"

    print("  [PASS] Stratified split test passed")


def test_image_augmentation():
    """Verify augmentation doesn't crash and preserves image properties"""
    print("Running test: Image augmentation...")
    from train_three_stage_hybrid_topk import apply_image_augmentation
    from PIL import Image
    import numpy as np

    # Create test image
    test_image = Image.new('RGB', (224, 224), color=(255, 0, 0))  # Red image

    # Apply augmentation
    aug_image = apply_image_augmentation(test_image, apply_aug=True)

    # Verify properties preserved
    assert aug_image.mode == 'RGB', f"Mode changed from RGB to {aug_image.mode}"
    assert aug_image.size == (224, 224), f"Size changed from (224, 224) to {aug_image.size}"
    assert isinstance(aug_image, Image.Image), f"Not a PIL Image: {type(aug_image)}"

    # Test that augmentation is actually applied (images should differ)
    # Run multiple times since augmentation is probabilistic
    differences = []
    for _ in range(10):
        aug = apply_image_augmentation(test_image, apply_aug=True)
        original_array = np.array(test_image)
        augmented_array = np.array(aug)
        diff = np.abs(original_array.astype(float) - augmented_array.astype(float)).sum()
        differences.append(diff)

    # At least some augmentations should modify the image
    assert max(differences) > 0, "Augmentation never changed the image"

    # Test disabling augmentation
    no_aug_image = apply_image_augmentation(test_image, apply_aug=False)
    assert np.array_equal(np.array(no_aug_image), np.array(test_image)), \
        "Augmentation applied when apply_aug=False"

    print("  [PASS] Image augmentation test passed")


def test_load_skincap_data():
    """Verify CSV loads and images exist (OPTIONAL - requires actual data)"""
    print("Running test: Data loading...")
    from train_three_stage_hybrid_topk import load_skincap_data, Config

    try:
        data = load_skincap_data()

        # Basic validation
        assert len(data) > 0, "No data loaded"
        assert all('image_path' in item for item in data), "Missing image_path"
        assert all('disease' in item for item in data), "Missing disease"
        assert all('caption' in item for item in data), "Missing caption"

        # Verify first image exists
        assert Path(data[0]['image_path']).exists(), "First image doesn't exist"

        print(f"  [PASS] Data loading test passed ({len(data)} samples loaded)")

        return True

    except FileNotFoundError as e:
        print(f"  [WARN] Skipping data loading test (CSV not found: {e})")
        return False


def run_all_tests():
    """Run all data pipeline tests"""
    print("=" * 70)
    print("RUNNING DATA PIPELINE TESTS")
    print("=" * 70)
    print()

    tests_passed = 0
    tests_failed = 0
    tests_skipped = 0

    # Core tests (always run)
    core_tests = [
        ("Fuzzy consolidation", test_fuzzy_consolidation),
        ("Group categorization", test_categorize_morphology),
        ("Top-K filtering", test_topk_filtering),
        ("Stratified split", test_stratified_split),
        ("Image augmentation", test_image_augmentation),
    ]

    for test_name, test_func in core_tests:
        try:
            test_func()
            tests_passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test_name} FAILED: {e}")
            tests_failed += 1
        except Exception as e:
            print(f"  [FAIL] {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1

    print()

    # Optional test (requires actual data)
    try:
        csv_exists = Path("./SkinCAP/skincap_v240623.csv").exists()
        if csv_exists:
            test_load_skincap_data()
            tests_passed += 1
        else:
            print("  [WARN] Skipping data loading test (CSV not found at ./SkinCAP/skincap_v240623.csv)")
            tests_skipped += 1
    except Exception as e:
        print(f"  [FAIL] Data loading ERROR: {e}")
        tests_failed += 1

    # Summary
    print()
    print("=" * 70)
    if tests_failed == 0:
        print("[SUCCESS] ALL TESTS PASSED!")
    else:
        print(f"[FAIL] SOME TESTS FAILED")
    print("=" * 70)
    print(f"Passed:  {tests_passed}")
    print(f"Failed:  {tests_failed}")
    print(f"Skipped: {tests_skipped}")
    print(f"Total:   {tests_passed + tests_failed + tests_skipped}")
    print("=" * 70)

    if tests_failed > 0:
        print("\n[WARN]️  Fix failing tests before running training!")
        exit(1)
    else:
        print("\n[SUCCESS] Data pipeline validated! Safe to proceed with training.")
        exit(0)


if __name__ == "__main__":
    run_all_tests()
