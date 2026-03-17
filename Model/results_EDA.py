"""
FuzzyTopK Classification Results - Error Analysis
Comprehensive visualization of inference results for both validation and whole dataset
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("FUZZY TOP-K ERROR ANALYSIS")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1] Loading results...")
results_dir = Path('./classification_results_fuzzytopk')

# Load metrics
with open(results_dir / 'results_fuzzytopk_val.json') as f:
    val_metrics = json.load(f)

with open(results_dir / 'results_fuzzytopk_whole.json') as f:
    whole_metrics = json.load(f)

# Load predictions
with open(results_dir / 'results_fuzzytopk_val_predictions.json') as f:
    val_preds_raw = json.load(f)
    val_preds = val_preds_raw['predictions'] if isinstance(val_preds_raw, dict) and 'predictions' in val_preds_raw else val_preds_raw

with open(results_dir / 'results_fuzzytopk_whole_predictions.json') as f:
    whole_preds_raw = json.load(f)
    whole_preds = whole_preds_raw['predictions'] if isinstance(whole_preds_raw, dict) and 'predictions' in whole_preds_raw else whole_preds_raw

print(f"✓ Loaded Val metrics: {len(val_metrics)} keys")
print(f"✓ Loaded Val predictions: {len(val_preds)} samples")
print(f"✓ Loaded Whole metrics: {len(whole_metrics)} keys")
print(f"✓ Loaded Whole predictions: {len(whole_preds)} samples")


# ============================================================================
# 1. OVERALL METRICS COMPARISON
# ============================================================================

print("\n[2] Comparing overall metrics...")

metrics_to_compare = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted', 'cohen_kappa']

comparison_data = []
for metric in metrics_to_compare:
    val_val = val_metrics.get(metric, 0)
    whole_val = whole_metrics.get(metric, 0)
    comparison_data.append({
        'Metric': metric.replace('_', ' ').title(),
        'Validation': val_val,
        'Whole': whole_val,
        'Diff': whole_val - val_val
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics_to_compare))
width = 0.35

val_scores = [val_metrics.get(m, 0) for m in metrics_to_compare]
whole_scores = [whole_metrics.get(m, 0) for m in metrics_to_compare]

ax.bar(x - width/2, val_scores, width, label='Validation', alpha=0.8)
ax.bar(x + width/2, whole_scores, width, label='Whole Dataset', alpha=0.8)

ax.set_ylabel('Score')
ax.set_title('Overall Metrics Comparison: Validation vs Whole Dataset')
ax.set_xticks(x)
ax.set_xticklabels([m.replace('_', '\n').title() for m in metrics_to_compare], rotation=0)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('./classification_results_fuzzytopk/01_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_metrics_comparison.png")
plt.close()


# ============================================================================
# 2. PER-CLASS ERROR RATE ANALYSIS
# ============================================================================

print("\n[3] Calculating per-class error rates...")

def calculate_per_class_errors(predictions):
    """Calculate error rate for each class"""
    errors_by_class = defaultdict(lambda: {'total': 0, 'errors': 0})

    for pred in predictions:
        gt = pred.get('ground_truth', '')
        predicted = pred.get('predicted', '')
        status = pred.get('status', '')

        if status == 'success' and predicted:
            errors_by_class[gt]['total'] += 1
            if predicted.lower() != gt.lower():
                errors_by_class[gt]['errors'] += 1

    # Calculate error rates
    error_rates = []
    for cls, stats in errors_by_class.items():
        if stats['total'] > 0:
            error_rate = stats['errors'] / stats['total']
            error_rates.append({
                'Class': cls,
                'Total': stats['total'],
                'Errors': stats['errors'],
                'Error_Rate': error_rate,
                'Accuracy': 1 - error_rate
            })

    return pd.DataFrame(error_rates).sort_values('Error_Rate', ascending=False)

val_errors = calculate_per_class_errors(val_preds)
whole_errors = calculate_per_class_errors(whole_preds)

print("\n=== VALIDATION SET - Per-Class Error Rates ===")
print(val_errors.to_string(index=False))

print("\n=== WHOLE DATASET - Per-Class Error Rates ===")
print(whole_errors.to_string(index=False))

# Visualize per-class error rates
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Validation
val_sorted = val_errors.sort_values('Error_Rate', ascending=True)
ax = axes[0]
colors = ['red' if x > 0.5 else 'orange' if x > 0.3 else 'green' for x in val_sorted['Error_Rate']]
ax.barh(val_sorted['Class'], val_sorted['Error_Rate'], color=colors, alpha=0.7)
ax.set_xlabel('Error Rate')
ax.set_title('Validation Set - Per-Class Error Rates')
ax.set_xlim([0, 1])
for i, v in enumerate(val_sorted['Error_Rate']):
    ax.text(v + 0.02, i, f'{v:.1%}', va='center')

# Whole
whole_sorted = whole_errors.sort_values('Error_Rate', ascending=True)
ax = axes[1]
colors = ['red' if x > 0.5 else 'orange' if x > 0.3 else 'green' for x in whole_sorted['Error_Rate']]
ax.barh(whole_sorted['Class'], whole_sorted['Error_Rate'], color=colors, alpha=0.7)
ax.set_xlabel('Error Rate')
ax.set_title('Whole Dataset - Per-Class Error Rates')
ax.set_xlim([0, 1])
for i, v in enumerate(whole_sorted['Error_Rate']):
    ax.text(v + 0.02, i, f'{v:.1%}', va='center')

plt.tight_layout()
plt.savefig('./classification_results_fuzzytopk/02_per_class_error_rates.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_per_class_error_rates.png")
plt.close()


# ============================================================================
# 3. CONFUSION MATRICES
# ============================================================================

print("\n[4] Creating confusion matrices...")

def create_confusion_matrix(predictions, class_list):
    """Create confusion matrix from predictions"""
    valid_preds = [p for p in predictions if p.get('status') == 'success' and p.get('predicted')]

    y_true = [p['ground_truth'] for p in valid_preds]
    y_pred = [p['predicted'] for p in valid_preds]

    # Create mapping
    class_to_idx = {cls: i for i, cls in enumerate(class_list)}

    y_true_idx = [class_to_idx.get(t, -1) for t in y_true]
    y_pred_idx = [class_to_idx.get(p, -1) for p in y_pred]

    # Filter valid pairs
    valid_pairs = [(t, p) for t, p in zip(y_true_idx, y_pred_idx) if t >= 0 and p >= 0]

    if not valid_pairs:
        return None

    y_true_valid = [p[0] for p in valid_pairs]
    y_pred_valid = [p[1] for p in valid_pairs]

    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=list(range(len(class_list))))
    return cm

# Get class list from metrics
top_classes = list(val_metrics.get('per_class_detailed', {}).keys())
if not top_classes:
    top_classes = sorted(set([p['ground_truth'] for p in val_preds if p.get('ground_truth')]))

print(f"Classes for confusion matrix: {len(top_classes)}")
print(f"Classes: {top_classes}")

# Create confusion matrices
val_cm = create_confusion_matrix(val_preds, top_classes)
whole_cm = create_confusion_matrix(whole_preds, top_classes)

if val_cm is not None and whole_cm is not None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Validation
    sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=top_classes, yticklabels=top_classes, cbar_kws={'label': 'Count'})
    axes[0].set_title('Validation Set - Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Ground Truth')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # Whole
    sns.heatmap(whole_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=top_classes, yticklabels=top_classes, cbar_kws={'label': 'Count'})
    axes[1].set_title('Whole Dataset - Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Ground Truth')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig('./classification_results_fuzzytopk/03_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 03_confusion_matrices.png")
    plt.close()
else:
    print("⚠ Could not create confusion matrices")


# ============================================================================
# 4. PER-CLASS DETAILED METRICS
# ============================================================================

print("\n[5] Extracting per-class medical metrics...")

val_per_class = val_metrics.get('per_class_detailed', {})
whole_per_class = whole_metrics.get('per_class_detailed', {})

if val_per_class:
    val_metrics_df = pd.DataFrame([
        {
            'Class': cls,
            'Sensitivity': stats.get('sensitivity', 0),
            'Specificity': stats.get('specificity', 0),
            'PPV': stats.get('ppv', 0),
            'NPV': stats.get('npv', 0),
            'F1': stats.get('f1_score', 0),
            'Support': stats.get('support', 0)
        }
        for cls, stats in val_per_class.items()
    ]).sort_values('Sensitivity', ascending=False)

    print("\n=== VALIDATION SET - Per-Class Medical Metrics ===")
    print(val_metrics_df.to_string(index=False))
else:
    print("Per-class metrics not available")

# Visualize per-class metrics
if val_per_class:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Sort by sensitivity for consistent ordering
    val_metrics_df_sorted = val_metrics_df.sort_values('Sensitivity')

    # Sensitivity
    ax = axes[0, 0]
    ax.barh(val_metrics_df_sorted['Class'], val_metrics_df_sorted['Sensitivity'], color='skyblue', alpha=0.7)
    ax.set_xlabel('Sensitivity (Recall)')
    ax.set_title('Validation - Sensitivity by Class')
    ax.set_xlim([0, 1])

    # Specificity
    ax = axes[0, 1]
    ax.barh(val_metrics_df_sorted['Class'], val_metrics_df_sorted['Specificity'], color='lightgreen', alpha=0.7)
    ax.set_xlabel('Specificity')
    ax.set_title('Validation - Specificity by Class')
    ax.set_xlim([0, 1])

    # PPV
    ax = axes[1, 0]
    ax.barh(val_metrics_df_sorted['Class'], val_metrics_df_sorted['PPV'], color='lightyellow', alpha=0.7)
    ax.set_xlabel('PPV (Precision)')
    ax.set_title('Validation - Precision (PPV) by Class')
    ax.set_xlim([0, 1])

    # F1 Score
    ax = axes[1, 1]
    ax.barh(val_metrics_df_sorted['Class'], val_metrics_df_sorted['F1'], color='lightcoral', alpha=0.7)
    ax.set_xlabel('F1 Score')
    ax.set_title('Validation - F1 Score by Class')
    ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig('./classification_results_fuzzytopk/04_per_class_medical_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 04_per_class_medical_metrics.png")
    plt.close()


# ============================================================================
# 5. TOP MISCLASSIFICATIONS
# ============================================================================

print("\n[6] Analyzing top misclassifications...")

def get_top_misclassifications(predictions, top_n=15):
    """Get the most common misclassification pairs"""
    misclassifications = defaultdict(int)

    for pred in predictions:
        gt = pred.get('ground_truth', '')
        predicted = pred.get('predicted', '')
        status = pred.get('status', '')

        if status == 'success' and predicted and predicted.lower() != gt.lower():
            key = f"{gt} → {predicted}"
            misclassifications[key] += 1

    # Sort and get top
    sorted_misclass = sorted(misclassifications.items(), key=lambda x: x[1], reverse=True)
    return sorted_misclass[:top_n]

val_misclass = get_top_misclassifications(val_preds, top_n=15)
whole_misclass = get_top_misclassifications(whole_preds, top_n=15)

print("\n=== TOP 15 MISCLASSIFICATIONS - VALIDATION SET ===")
for i, (pair, count) in enumerate(val_misclass, 1):
    print(f"{i:2d}. {pair}: {count} times")

print("\n=== TOP 15 MISCLASSIFICATIONS - WHOLE DATASET ===")
for i, (pair, count) in enumerate(whole_misclass, 1):
    print(f"{i:2d}. {pair}: {count} times")

# Visualize top misclassifications
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Validation
if val_misclass:
    pairs, counts = zip(*val_misclass)
    ax = axes[0]
    ax.barh(range(len(pairs)), counts, color='salmon', alpha=0.7)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=9)
    ax.set_xlabel('Frequency')
    ax.set_title('Validation Set - Top 15 Misclassifications')
    for i, count in enumerate(counts):
        ax.text(count + 0.1, i, str(count), va='center')

# Whole
if whole_misclass:
    pairs, counts = zip(*whole_misclass)
    ax = axes[1]
    ax.barh(range(len(pairs)), counts, color='salmon', alpha=0.7)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=9)
    ax.set_xlabel('Frequency')
    ax.set_title('Whole Dataset - Top 15 Misclassifications')
    for i, count in enumerate(counts):
        ax.text(count + 0.1, i, str(count), va='center')

plt.tight_layout()
plt.savefig('./classification_results_fuzzytopk/05_top_misclassifications.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_top_misclassifications.png")
plt.close()


# ============================================================================
# 6. SAMPLE STATISTICS
# ============================================================================

print("\n[7] Calculating statistics...")

def get_statistics(predictions):
    total = len(predictions)
    successful = sum(1 for p in predictions if p.get('status') == 'success')
    failed = total - successful

    class_distribution = Counter([p.get('ground_truth', 'unknown') for p in predictions if p.get('ground_truth')])

    return {
        'Total Samples': total,
        'Successful': successful,
        'Failed': failed,
        'Success Rate': f"{100*successful/total:.1f}%",
        'Unique Classes': len(class_distribution),
        'Class Distribution': class_distribution
    }

val_stats = get_statistics(val_preds)
whole_stats = get_statistics(whole_preds)

print("\n=== VALIDATION SET STATISTICS ===")
for key, value in val_stats.items():
    if key != 'Class Distribution':
        print(f"{key}: {value}")

print("\n=== WHOLE DATASET STATISTICS ===")
for key, value in whole_stats.items():
    if key != 'Class Distribution':
        print(f"{key}: {value}")

# Visualize class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Validation
val_dist = val_stats['Class Distribution']
classes, counts = zip(*sorted(val_dist.items(), key=lambda x: x[1], reverse=True))
ax = axes[0]
ax.barh(classes, counts, color='steelblue', alpha=0.7)
ax.set_xlabel('Sample Count')
ax.set_title(f'Validation Set - Class Distribution (n={val_stats["Total Samples"]})')
for i, count in enumerate(counts):
    ax.text(count + 0.5, i, str(count), va='center')

# Whole
whole_dist = whole_stats['Class Distribution']
classes, counts = zip(*sorted(whole_dist.items(), key=lambda x: x[1], reverse=True))
ax = axes[1]
ax.barh(classes, counts, color='steelblue', alpha=0.7)
ax.set_xlabel('Sample Count')
ax.set_title(f'Whole Dataset - Class Distribution (n={whole_stats["Total Samples"]})')
for i, count in enumerate(counts):
    ax.text(count + 0.5, i, str(count), va='center')

plt.tight_layout()
plt.savefig('./classification_results_fuzzytopk/06_class_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_class_distribution.png")
plt.close()


# ============================================================================
# 7. SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("FUZZY TOP-K ERROR ANALYSIS SUMMARY")
print("="*80)

print("\nVALIDATION SET:")
print(f"  Total Samples:        {val_metrics.get('total_samples', 'N/A')}")
print(f"  Accuracy:             {val_metrics.get('accuracy', 0):.4f}")
print(f"  Balanced Accuracy:    {val_metrics.get('balanced_accuracy', 0):.4f}")
print(f"  F1 (Macro):           {val_metrics.get('f1_macro', 0):.4f}")
print(f"  F1 (Weighted):        {val_metrics.get('f1_weighted', 0):.4f}")
print(f"  Cohen's Kappa:        {val_metrics.get('cohen_kappa', 0):.4f}")
print(f"  Matthews Corr. Coef:  {val_metrics.get('matthews_corrcoef', 0):.4f}")

print(f"\n  Best Performing Classes (Val):")
for i, row in val_errors.nsmallest(3, 'Error_Rate').iterrows():
    print(f"    {row['Class']}: {row['Error_Rate']:.1%} error rate")

print(f"\n  Worst Performing Classes (Val):")
for i, row in val_errors.nlargest(3, 'Error_Rate').iterrows():
    print(f"    {row['Class']}: {row['Error_Rate']:.1%} error rate")

print(f"\n\nWHOLE DATASET:")
print(f"  Total Samples:        {whole_metrics.get('total_samples', 'N/A')}")
print(f"  Accuracy:             {whole_metrics.get('accuracy', 0):.4f}")
print(f"  Balanced Accuracy:    {whole_metrics.get('balanced_accuracy', 0):.4f}")
print(f"  F1 (Macro):           {whole_metrics.get('f1_macro', 0):.4f}")
print(f"  F1 (Weighted):        {whole_metrics.get('f1_weighted', 0):.4f}")
print(f"  Cohen's Kappa:        {whole_metrics.get('cohen_kappa', 0):.4f}")
print(f"  Matthews Corr. Coef:  {whole_metrics.get('matthews_corrcoef', 0):.4f}")

print(f"\n  Best Performing Classes (Whole):")
for i, row in whole_errors.nsmallest(3, 'Error_Rate').iterrows():
    print(f"    {row['Class']}: {row['Error_Rate']:.1%} error rate")

print(f"\n  Worst Performing Classes (Whole):")
for i, row in whole_errors.nlargest(3, 'Error_Rate').iterrows():
    print(f"    {row['Class']}: {row['Error_Rate']:.1%} error rate")

print(f"\n\nKEY INSIGHTS:")
diff_accuracy = whole_metrics.get('accuracy', 0) - val_metrics.get('accuracy', 0)
if diff_accuracy > 0.05:
    print(f"  ⚠ Generalization gap detected: Whole dataset {diff_accuracy:.1%} {'better' if diff_accuracy > 0 else 'worse'} than validation")
elif diff_accuracy < -0.05:
    print(f"  ⚠ Possible overfitting: Whole dataset {abs(diff_accuracy):.1%} worse than validation")
else:
    print(f"  ✓ Good generalization: Similar performance across val and whole datasets")

print(f"\n  Generated visualizations:")
print(f"    1. 01_metrics_comparison.png")
print(f"    2. 02_per_class_error_rates.png")
print(f"    3. 03_confusion_matrices.png")
print(f"    4. 04_per_class_medical_metrics.png")
print(f"    5. 05_top_misclassifications.png")
print(f"    6. 06_class_distribution.png")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE")
print("="*80)
