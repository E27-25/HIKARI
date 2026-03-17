import pandas as pd
import numpy as np
from collections import defaultdict

# Dataset distribution from the provided data
dataset_stats = {
    "1. Pigmented Lesions": {
        "total_rows": 506,
        "num_diseases": 30,
        "top_diseases": [
            ("melanocytic nevi", 119),
            ("melanoma", 57),
            ("urticaria pigmentosa", 25),
            ("lentigo maligna", 23),
            ("epidermal nevus", 21),
        ]
    },
    "2. Papules, Nodules & Tumors": {
        "total_rows": 1009,
        "num_diseases": 55,
        "top_diseases": [
            ("squamous cell carcinoma", 138),
            ("basal cell carcinoma", 131),
            ("sarcoidosis", 75),
            ("seborrheic keratosis", 72),
            ("granuloma annulare", 42),
        ]
    },
    "3. Papulosquamous/Scaly Plaques": {
        "total_rows": 910,
        "num_diseases": 30,
        "top_diseases": [
            ("psoriasis", 127),
            ("lichen planus", 92),
            ("allergic contact dermatitis", 70),
            ("mycosis fungoides", 68),
            ("pityriasis rubra pilaris", 48),
        ]
    },
    "4. Vesicles, Pustules & Acneiform": {
        "total_rows": 459,
        "num_diseases": 16,
        "top_diseases": [
            ("acne vulgaris", 76),
            ("neutrophilic dermatoses", 69),
            ("folliculitis", 60),
            ("acne", 46),
            ("hailey hailey disease", 35),
        ]
    },
    "5. Erythema & Urticaria": {
        "total_rows": 590,
        "num_diseases": 22,
        "top_diseases": [
            ("lupus erythematosus", 95),
            ("photodermatoses", 76),
            ("erythema multiforme", 50),
            ("drug eruption", 48),
            ("pyogenic granuloma", 37),
        ]
    },
    "6. Sclerosis, Atrophy & Depigmentation": {
        "total_rows": 251,
        "num_diseases": 11,
        "top_diseases": [
            ("scleroderma", 81),
            ("vitiligo", 38),
            ("tuberous sclerosis", 29),
            ("necrobiosis lipoidica", 25),
            ("ehlers danlos syndrome", 19),
        ]
    },
    "7. Infections & Infestations": {
        "total_rows": 275,
        "num_diseases": 14,
        "top_diseases": [
            ("scabies", 70),
            ("verruca vulgaris", 50),
            ("nematode infection", 43),
            ("lyme disease", 27),
            ("pediculosis lids", 26),
        ]
    }
}

total_dataset = 4000

print("="*80)
print("ANALYSIS: Optimal Top-K for 3-Stage VLM Training (Qwen3VL-8B)")
print("="*80)
print("\nDataset Overview:")
print(f"Total samples: {total_dataset}")
print(f"Total groups: {len(dataset_stats)}")
print(f"Total unique diseases: {sum(g['num_diseases'] for g in dataset_stats.values())}")
print("\n" + "="*80)

# Calculate statistics for each group
print("\n📊 GROUP-LEVEL STATISTICS:\n")
for group_name, stats in dataset_stats.items():
    pct = (stats['total_rows'] / total_dataset) * 100
    avg_samples = stats['total_rows'] / stats['num_diseases']
    
    print(f"\n{group_name}")
    print(f"  Total Samples: {stats['total_rows']:4d} ({pct:5.2f}%)")
    print(f"  Diseases: {stats['num_diseases']:2d}")
    print(f"  Avg samples/disease: {avg_samples:.1f}")
    
    # Calculate concentration in top diseases
    top5_sum = sum(count for _, count in stats['top_diseases'])
    top5_pct = (top5_sum / stats['total_rows']) * 100
    print(f"  Top-5 concentration: {top5_pct:.1f}%")

print("\n" + "="*80)
print("\n🎯 RECOMMENDED TOP-K VALUES FOR BALANCED TRAINING:\n")

# Strategy 1: Coverage-based (target 70-80% coverage)
print("\n--- Strategy 1: Coverage-Based (70-80% target) ---")
print("Goal: Capture majority of samples while maintaining balance\n")

for group_name, stats in dataset_stats.items():
    diseases = stats['top_diseases']
    cumulative = 0
    k = 0
    
    for i, (disease, count) in enumerate(diseases, 1):
        cumulative += count
        coverage = (cumulative / stats['total_rows']) * 100
        if coverage >= 70 and k == 0:
            k = i
    
    if k == 0:
        k = min(5, stats['num_diseases'])
    
    top_k_coverage = sum(c for _, c in diseases[:k]) / stats['total_rows'] * 100
    
    print(f"{group_name}")
    print(f"  Recommended K = {k}")
    print(f"  Coverage: {top_k_coverage:.1f}%")
    print(f"  Samples included: {sum(c for _, c in diseases[:k])}/{stats['total_rows']}")

print("\n" + "-"*80)

# Strategy 2: Balanced samples (target ~100-150 samples per group for 2nd stage)
print("\n--- Strategy 2: Sample-Balanced (100-150 samples target) ---")
print("Goal: Ensure similar training samples per group for 2nd stage\n")

target_samples = 120  # Target samples per group for balanced 2nd stage

for group_name, stats in dataset_stats.items():
    diseases = stats['top_diseases']
    cumulative = 0
    k = 0
    
    for i, (disease, count) in enumerate(diseases, 1):
        cumulative += count
        if cumulative >= target_samples:
            k = i
            break
    
    if k == 0:
        k = min(len(diseases), stats['num_diseases'])
        cumulative = sum(c for _, c in diseases[:k])
    
    print(f"{group_name}")
    print(f"  Recommended K = {k}")
    print(f"  Samples: {cumulative}")
    print(f"  Coverage: {(cumulative/stats['total_rows'])*100:.1f}%")

print("\n" + "-"*80)

# Strategy 3: Hybrid - consider both coverage and minimum samples
print("\n--- Strategy 3: RECOMMENDED HYBRID APPROACH ---")
print("Goal: Balance coverage + ensure minimum samples per disease\n")

min_samples_per_disease = 15  # Minimum samples to be useful for training

recommended_k = {}

for group_name, stats in dataset_stats.items():
    diseases = stats['top_diseases']
    
    # Find K where:
    # 1. Coverage >= 65-70%
    # 2. Last included disease has >= min_samples
    # 3. Or take all diseases if group is small
    
    if stats['num_diseases'] <= 10:
        # Small group - take more diseases
        k = min(8, stats['num_diseases'])
    else:
        cumulative = 0
        k = 0
        for i, (disease, count) in enumerate(diseases, 1):
            cumulative += count
            coverage = (cumulative / stats['total_rows']) * 100
            
            if coverage >= 70 or i >= 10:  # Stop at 70% or max 10 diseases
                if count >= min_samples_per_disease or i <= 5:
                    k = i
                break
        
        if k == 0:
            k = 5  # Default minimum
    
    # Calculate final stats
    included_samples = sum(c for _, c in diseases[:k])
    coverage = (included_samples / stats['total_rows']) * 100
    
    recommended_k[group_name] = {
        'k': k,
        'samples': included_samples,
        'coverage': coverage,
        'total': stats['total_rows']
    }
    
    print(f"{group_name}")
    print(f"  ✓ Top-K = {k}")
    print(f"  ✓ Samples: {included_samples}/{stats['total_rows']} ({coverage:.1f}%)")
    print(f"  ✓ Remaining: {stats['total_rows'] - included_samples} samples in 'Other' category")
    print()

print("="*80)
print("\n📋 FINAL RECOMMENDATIONS FOR 3-STAGE TRAINING:\n")

print("Stage 1 (Group Classification):")
print("  - Use ALL 7 groups")
print("  - Samples per group: as shown above")
print("  - Total: 4000 samples")
print()

print("Stage 2 (Disease Classification within Groups):")
stage2_total = sum(v['samples'] for v in recommended_k.values())
print(f"  - Use Top-K diseases per group (K values above)")
print(f"  - Total samples for top diseases: {stage2_total}")
print(f"  - Create 'Other' category per group for remaining diseases")
print()

# Calculate balance score
samples = [v['samples'] for v in recommended_k.values()]
balance_score = min(samples) / max(samples) * 100

print("Stage 3 (Caption Generation):")
print("  - Use model from Stage 2 (last epoch)")
print("  - Fine-tune on all 4000 samples with captions")
print()

print(f"Balance Score: {balance_score:.1f}% (higher is better)")
print(f"  Min samples: {min(samples)} (Group: {min(recommended_k.items(), key=lambda x: x[1]['samples'])[0]})")
print(f"  Max samples: {max(samples)} (Group: {max(recommended_k.items(), key=lambda x: x[1]['samples'])[0]})")
print()

# Export configuration
print("="*80)
print("\n💾 CONFIGURATION FOR TRAINING SCRIPT:\n")
print("config = {")
for group_name, config in recommended_k.items():
    print(f"    '{group_name}': {config['k']},  # {config['samples']} samples ({config['coverage']:.1f}%)")
print("}")

print("\n" + "="*80)
print("\n⚠️  IMPORTANT CONSIDERATIONS:\n")
print("1. Class Imbalance:")
print("   - Within groups: Use weighted loss or focal loss")
print("   - Between groups: Current distribution is acceptable")
print()
print("2. 'Other' Category Strategy:")
print("   - Option A: Create 'Other' bin per group (recommended)")
print("   - Option B: Exclude rare diseases (may lose data)")
print()
print("3. Data Augmentation:")
print("   - Apply to minority classes within each group")
print("   - Consider for groups with <500 samples")
print()
print("4. Validation Split:")
print("   - Stratify by both group AND disease")
print("   - Recommend 80-10-10 (train-val-test)")
print()
print("5. Evaluation Metrics:")
print("   - Stage 1: Accuracy, F1-macro across 7 groups")
print("   - Stage 2: Per-group F1, overall weighted F1")
print("   - Stage 3: BLEU, ROUGE, CIDEr for captions")

print("\n" + "="*80)
