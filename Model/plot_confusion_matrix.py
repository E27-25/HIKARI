"""
Generate Confusion Matrix from Classification Results JSON

Usage:
    python plot_confusion_matrix.py
    python plot_confusion_matrix.py --json ./classification_results/classification_results_val.json
    python plot_confusion_matrix.py --json ./classification_results/classification_results_val.json --top 30
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import argparse


def load_results(json_path: str) -> dict:
    """Load results from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_disease_name(name: str) -> str:
    """Normalize disease name for comparison"""
    if not name:
        return ""
    name = name.lower().strip()
    name = name.replace("-", " ").replace("_", " ")
    name = " ".join(name.split())
    return name


def match_disease_names(pred: str, ground_truth: str) -> bool:
    """
    Check if prediction matches ground truth with flexible matching.
    
    Handles cases like:
    - "melanocytic-nevi" matches "melanocytic nevi" (hyphen vs space)
    - "seborrheic-keratosis-irritated" matches "seborrheic keratosis" (partial match)
    """
    pred_norm = normalize_disease_name(pred)
    gt_norm = normalize_disease_name(ground_truth)
    
    # Exact match after normalization
    if pred_norm == gt_norm:
        return True
    
    # Partial match: check if one contains all words of the other
    pred_words = set(pred_norm.split())
    gt_words = set(gt_norm.split())
    
    # If ground truth words are subset of prediction
    if gt_words.issubset(pred_words) and len(gt_words) >= 2:
        return True
    
    # If prediction words are subset of ground truth
    if pred_words.issubset(gt_words) and len(pred_words) >= 2:
        return True
    
    return False


def find_canonical_name(name: str, all_classes: set) -> str:
    """
    Find the canonical (shortest matching) name from the class set.
    E.g., "seborrheic keratosis irritated" -> "seborrheic keratosis" if that exists.
    """
    name_norm = normalize_disease_name(name)
    name_words = set(name_norm.split())
    
    best_match = name_norm
    best_len = len(name_words)
    
    for cls in all_classes:
        cls_norm = normalize_disease_name(cls)
        cls_words = set(cls_norm.split())
        
        # Check if this class is a subset (shorter version) of our name
        if cls_words.issubset(name_words) and len(cls_words) < best_len:
            best_match = cls_norm
            best_len = len(cls_words)
        # Or if our name is subset of this class
        elif name_words.issubset(cls_words) and len(name_words) == best_len:
            # Prefer existing class names
            if cls in all_classes:
                best_match = cls_norm
    
    return best_match


def build_confusion_matrix(predictions: list) -> tuple:
    """Build confusion matrix from predictions with flexible matching"""
    
    # First pass: collect all raw class names from GROUND TRUTH only
    gt_classes = set()
    for pred in predictions:
        if pred.get("status") == "success":
            gt = pred.get("ground_truth", "")
            if gt:
                gt_classes.add(normalize_disease_name(gt))
    
    # Create canonical class list from ground truth
    all_classes = sorted(gt_classes)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    
    # Helper: find best matching class for a prediction
    def find_best_match(pred_name: str) -> str:
        pred_norm = normalize_disease_name(pred_name)
        pred_words = set(pred_norm.split())
        
        # Exact match
        if pred_norm in class_to_idx:
            return pred_norm
        
        # Partial match: find class where pred contains all class words OR class contains all pred words
        best_match = None
        best_score = 0
        
        for cls in all_classes:
            cls_words = set(cls.split())
            
            # Case 1: prediction contains all ground truth words (e.g., "seborrheic keratosis irritated" contains "seborrheic keratosis")
            if cls_words.issubset(pred_words) and len(cls_words) >= 2:
                score = len(cls_words) * 2  # Prefer longer matches
                if score > best_score:
                    best_score = score
                    best_match = cls
            
            # Case 2: ground truth contains all prediction words
            elif pred_words.issubset(cls_words) and len(pred_words) >= 2:
                score = len(pred_words)
                if score > best_score:
                    best_score = score
                    best_match = cls
        
        return best_match if best_match else pred_norm
    
    # Build confusion matrix
    n_classes = len(all_classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    matched_count = 0
    total_count = 0
    match_details = []  # Track what got matched
    
    for pred in predictions:
        if pred.get("status") == "success":
            gt_raw = pred.get("ground_truth", "")
            pr_raw = pred.get("predicted", "")
            
            gt_norm = normalize_disease_name(gt_raw)
            pr_matched = find_best_match(pr_raw)
            
            if gt_norm in class_to_idx:
                total_count += 1
                
                if pr_matched in class_to_idx:
                    cm[class_to_idx[gt_norm], class_to_idx[pr_matched]] += 1
                    
                    if gt_norm == pr_matched:
                        matched_count += 1
                        if normalize_disease_name(pr_raw) != gt_norm:
                            match_details.append(f"   ✓ '{pr_raw}' → '{gt_norm}'")
    
    accuracy = matched_count / total_count * 100 if total_count > 0 else 0
    print(f"   Flexible matching accuracy: {matched_count}/{total_count} ({accuracy:.2f}%)")
    
    if match_details[:5]:
        print("   Partial matches applied:")
        for detail in match_details[:5]:
            print(detail)
        if len(match_details) > 5:
            print(f"   ... and {len(match_details) - 5} more")
    
    return cm, all_classes


def plot_confusion_matrix(cm: np.ndarray, labels: list, output_path: str, 
                          title: str = "Confusion Matrix", top_n: int = None,
                          figsize: tuple = None, normalize: bool = True):
    """Plot and save confusion matrix"""
    
    # Filter to top N classes by sample count if specified
    if top_n and len(labels) > top_n:
        class_counts = cm.sum(axis=1)
        top_indices = np.argsort(class_counts)[-top_n:]
        top_indices = sorted(top_indices)  # Keep alphabetical order within top
        
        cm = cm[np.ix_(top_indices, top_indices)]
        labels = [labels[i] for i in top_indices]
        title = f"{title} (Top {top_n} classes)"
    
    n_classes = len(labels)
    
    # Auto figure size
    if figsize is None:
        size = max(12, n_classes * 0.35)
        figsize = (size, size)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize if requested
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm.astype(float), row_sums, 
                               out=np.zeros_like(cm, dtype=float), 
                               where=row_sums != 0)
        fmt = '.2f'
        vmin, vmax = 0, 1
    else:
        cm_display = cm
        fmt = 'd'
        vmin, vmax = 0, cm.max()
    
    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=n_classes <= 25,  # Show numbers only if not too many classes
        fmt=fmt,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    
    # Rotate labels
    plt.xticks(rotation=90, ha='center', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {output_path}")
    return output_path


def plot_top_confusions(predictions: list, output_path: str, top_n: int = 20):
    """Plot top confusion pairs as bar chart"""
    
    confusion_counts = defaultdict(int)
    
    for pred in predictions:
        if pred.get("status") == "success":
            gt = normalize_disease_name(pred.get("ground_truth", ""))
            pr = normalize_disease_name(pred.get("predicted", ""))
            
            if gt and pr and gt != pr:
                confusion_counts[(gt, pr)] += 1
    
    if not confusion_counts:
        print("No confusions found!")
        return None
    
    # Sort by count
    sorted_confusions = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_confusions) * 0.4)))
    
    labels = [f"{gt} → {pr}" for (gt, pr), _ in sorted_confusions]
    counts = [c for _, c in sorted_confusions]
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(counts)))
    
    bars = ax.barh(range(len(labels)), counts, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title(f'Top {top_n} Confusion Pairs', fontsize=14, fontweight='bold')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                str(count), va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Top confusions chart saved to: {output_path}")
    return output_path


def plot_per_class_accuracy(cm: np.ndarray, labels: list, output_path: str, top_n: int = 30):
    """Plot per-class accuracy as horizontal bar chart"""
    
    # Calculate per-class accuracy
    row_sums = cm.sum(axis=1)
    diagonal = np.diag(cm)
    
    # Only include classes with samples
    valid_mask = row_sums > 0
    accuracies = []
    class_labels = []
    supports = []
    
    for i, (label, acc, total) in enumerate(zip(labels, diagonal, row_sums)):
        if total > 0:
            accuracies.append(acc / total)
            class_labels.append(f"{label} (n={int(total)})")
            supports.append(total)
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)
    
    # Take bottom and top classes
    if len(sorted_indices) > top_n:
        # Show worst and best performers
        half = top_n // 2
        indices_to_show = list(sorted_indices[:half]) + list(sorted_indices[-half:])
    else:
        indices_to_show = sorted_indices
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(indices_to_show) * 0.35)))
    
    show_labels = [class_labels[i] for i in indices_to_show]
    show_accs = [accuracies[i] for i in indices_to_show]
    
    colors = plt.cm.RdYlGn(show_accs)  # Red for low, green for high
    
    bars = ax.barh(range(len(show_labels)), show_accs, color=colors)
    ax.set_yticks(range(len(show_labels)))
    ax.set_yticklabels(show_labels, fontsize=8)
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add accuracy labels
    for bar, acc in zip(bars, show_accs):
        ax.text(min(acc + 0.02, 0.95), bar.get_y() + bar.get_height()/2, 
                f'{acc:.1%}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Per-class accuracy chart saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Plot confusion matrix from JSON results')
    parser.add_argument('--json', type=str, 
                        default='./classification_results/classification_results_val.json',
                        help='Path to JSON results file')
    parser.add_argument('--output_dir', type=str, default='./classification_results',
                        help='Output directory for plots')
    parser.add_argument('--top', type=int, default=30,
                        help='Number of top classes to show (default: 30)')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Show raw counts instead of normalized values')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.json}")
    results = load_results(args.json)
    
    predictions = results.get("predictions", [])
    split_name = results.get("split_name", "unknown")
    
    print(f"Found {len(predictions)} predictions")
    
    # Calculate overall accuracy with flexible matching FIRST
    print("\n" + "="*60)
    print("OVERALL ACCURACY SUMMARY (with flexible matching)")
    print("="*60)
    
    correct = 0
    total = 0
    correct_examples = []
    incorrect_examples = []
    
    for pred in predictions:
        if pred.get("status") == "success":
            gt = pred.get("ground_truth", "")
            pr = pred.get("predicted", "")
            
            total += 1
            if match_disease_names(pr, gt):
                correct += 1
                if normalize_disease_name(pr) != normalize_disease_name(gt):
                    correct_examples.append(f"'{pr}' ≈ '{gt}'")
            else:
                incorrect_examples.append(f"'{pr}' ≠ '{gt}'")
    
    accuracy = correct / total * 100 if total > 0 else 0
    
    print(f"\n📊 Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"\n   ✓ Correct:   {correct}")
    print(f"   ✗ Incorrect: {total - correct}")
    
    if correct_examples[:5]:
        print(f"\n   Partial matches counted as correct:")
        for ex in correct_examples[:5]:
            print(f"      {ex}")
        if len(correct_examples) > 5:
            print(f"      ... and {len(correct_examples) - 5} more")
    
    # Build confusion matrix
    print("\n" + "="*60)
    print("Building confusion matrix...")
    cm, labels = build_confusion_matrix(predictions)
    print(f"Found {len(labels)} unique classes")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot confusion matrix (full)
    plot_confusion_matrix(
        cm, labels,
        output_path=str(output_dir / f"confusion_matrix_{split_name}_full.png"),
        title=f"Confusion Matrix - {split_name.upper()} (All {len(labels)} classes)",
        normalize=not args.no_normalize
    )
    
    # Plot confusion matrix (top N)
    if len(labels) > args.top:
        plot_confusion_matrix(
            cm, labels,
            output_path=str(output_dir / f"confusion_matrix_{split_name}_top{args.top}.png"),
            title=f"Confusion Matrix - {split_name.upper()}",
            top_n=args.top,
            normalize=not args.no_normalize
        )
    
    # Plot top confusions
    plot_top_confusions(
        predictions,
        output_path=str(output_dir / f"top_confusions_{split_name}.png"),
        top_n=20
    )
    
    # Plot per-class accuracy
    plot_per_class_accuracy(
        cm, labels,
        output_path=str(output_dir / f"per_class_accuracy_{split_name}.png"),
        top_n=args.top
    )
    
    # Save overall summary
    summary = {
        "split": split_name,
        "total_predictions": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "accuracy_percent": f"{accuracy:.2f}%",
        "num_classes": len(labels),
        "partial_matches_as_correct": len(correct_examples)
    }
    
    summary_path = output_dir / f"accuracy_summary_{split_name}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ All plots saved to: {output_dir}")
    print(f"   Summary saved to: {summary_path}")
    
    # Final summary box
    print("\n" + "="*60)
    print(f"  FINAL ACCURACY: {accuracy:.2f}%  ({correct}/{total})")
    print("="*60)


if __name__ == "__main__":
    main()
