"""
Inference script for Stage 1 Group Classification Model (3-Stage Pipeline)
Strategy G: Top-15 diseases, 3 or 4 groups
Based on Qwen3-VL-8B fine-tuned for skin disease group classification

Two flows:
- Flow 1: Test on validation set from training split
- Flow 2: Test on whole dataset

IMPORTANT: Set GROUP_MODE below to match the training GROUP_MODE used.
"""

import os
os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import pandas as pd
import numpy as np
import random
import json
import re
from pathlib import Path
from PIL import Image
from collections import Counter
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from thefuzz import process as fuzz_process

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score, multilabel_confusion_matrix,
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False


# ============================================================================
# CONFIG
# ============================================================================

# -----------------------------------------------------------------------
# Strategy G: Set GROUP_MODE to match training (must be identical)
# "4group": Inflammatory / Benign / Malignant / Acne
# "3group": Inflammatory / Benign+Acne merged / Malignant
# -----------------------------------------------------------------------
# GROUP_MODE = "3group" # "4group" or "3group" (set at module level for easy access in functions)
# TOP_N = 15           # 15 = top-15 diseases, 10 = top-10 diseases (must match training)

# GROUP_MODE = "3group"
# TOP_N = 10

GROUP_MODE = "4group"
TOP_N = 10


@dataclass
class InferenceConfig:
    SEED: int = 42
    MODEL_PATH_MERGED: str = f"./skincap_3stage_group_{GROUP_MODE}_top{TOP_N}_merged"
    CSV_PATH: str = "./SkinCAP/skincap_v240623.csv"
    IMAGE_BASE_PATH: str = "./SkinCAP/skincap"
    SPLIT_INFO_PATH: str = "./split_info_3stage.json"
    BATCH_SIZE: int = 8
    MAX_NEW_TOKENS: int = 32   # Group name is ~8 tokens; 128 caused looping output
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.9
    FUZZY_THRESHOLD: int = 91
    OUTPUT_DIR: str = "./group_classification_results"


# Group names — Strategy G (must match training GROUP_MODE)
_GROUP_NAMES_4 = [
    '1. Inflammatory & Autoimmune Diseases',
    '2. Benign Tumors, Nevi & Cysts',
    '3. Malignant Skin Tumors',
    '4. Acne & Follicular Disorders',
]
_GROUP_NAMES_3 = [
    '1. Inflammatory & Autoimmune Diseases',
    '2. Benign & Other Non-Malignant',
    '3. Malignant Skin Tumors',
]
GROUP_NAMES = _GROUP_NAMES_4 if GROUP_MODE == "4group" else _GROUP_NAMES_3


# ============================================================================
# SEED & UTILS
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def norm_disease(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.lower().strip().replace("-", " ")


def fuzzy_consolidate_diseases(data: List[Dict], threshold: int = 91) -> List[Dict]:
    for item in data:
        item["disease_original"] = item["disease"]
        item["disease"] = norm_disease(item["disease"])

    unique_diseases = sorted(set(item["disease"] for item in data))
    mapping = {}
    processed = set()

    for disease in unique_diseases:
        if disease in processed:
            continue
        matches = fuzz_process.extract(disease, unique_diseases, limit=None)
        close_matches = [m[0] for m in matches if m[1] >= threshold]
        for match in close_matches:
            mapping[match] = disease
            processed.add(match)

    # Force SCC variants → SCCIS regardless of fuzzy grouping result
    FORCED_CANONICAL = {"squamous cell carcinoma": "squamous cell carcinoma in situ"}
    for disease in list(mapping.keys()):
        if mapping[disease] in FORCED_CANONICAL:
            mapping[disease] = FORCED_CANONICAL[mapping[disease]]
    for src, tgt in FORCED_CANONICAL.items():
        mapping[src] = tgt

    for item in data:
        item["disease"] = mapping.get(item["disease"], item["disease"])

    return data


# Top 15 diseases — Strategy G (must match training code)
TOP_15_DISEASES = {
    "squamous cell carcinoma", "basal cell carcinoma", "psoriasis",
    "melanocytic nevi", "lupus erythematosus", "lichen planus",
    "scleroderma", "photodermatoses", "acne vulgaris", "sarcoidosis",
    "seborrheic keratosis", "allergic contact dermatitis",
    "neutrophilic dermatoses", "mycosis fungoides", "folliculitis",
}

# Top 10 diseases (top-15 minus positions 11-15)
TOP_10_DISEASES = {
    "squamous cell carcinoma", "basal cell carcinoma", "psoriasis",
    "melanocytic nevi", "lupus erythematosus", "lichen planus",
    "scleroderma", "photodermatoses", "acne vulgaris", "sarcoidosis",
}

# Disease groups — Strategy G (must match training GROUP_MODE)
_DISEASE_GROUPS_4 = {
    "1. Inflammatory & Autoimmune Diseases": [
        "psoriasis", "lupus erythematosus", "lichen planus", "scleroderma",
        "photodermatoses", "sarcoidosis", "allergic contact dermatitis",
        "neutrophilic dermatoses",
    ],
    "2. Benign Tumors, Nevi & Cysts": [
        "melanocytic nevi", "seborrheic keratosis",
    ],
    "3. Malignant Skin Tumors": [
        "squamous cell carcinoma in situ", "basal cell carcinoma", "mycosis fungoides",
    ],
    "4. Acne & Follicular Disorders": [
        "acne vulgaris", "folliculitis",
    ],
}
_DISEASE_GROUPS_3 = {
    "1. Inflammatory & Autoimmune Diseases": [
        "psoriasis", "lupus erythematosus", "lichen planus", "scleroderma",
        "photodermatoses", "sarcoidosis", "allergic contact dermatitis",
        "neutrophilic dermatoses",
    ],
    "2. Benign & Other Non-Malignant": [
        "melanocytic nevi", "seborrheic keratosis",
        "acne vulgaris", "folliculitis",
    ],
    "3. Malignant Skin Tumors": [
        "squamous cell carcinoma in situ", "basal cell carcinoma", "mycosis fungoides",
    ],
}
_DISEASE_GROUPS_4_TOP10 = {
    "1. Inflammatory & Autoimmune Diseases": [
        "psoriasis", "lupus erythematosus", "lichen planus", "scleroderma",
        "photodermatoses", "sarcoidosis",
    ],
    "2. Benign Tumors, Nevi & Cysts": ["melanocytic nevi"],
    "3. Malignant Skin Tumors": ["squamous cell carcinoma in situ", "basal cell carcinoma"],
    "4. Acne & Follicular Disorders": ["acne vulgaris"],
}
_DISEASE_GROUPS_3_TOP10 = {
    "1. Inflammatory & Autoimmune Diseases": [
        "psoriasis", "lupus erythematosus", "lichen planus", "scleroderma",
        "photodermatoses", "sarcoidosis",
    ],
    "2. Benign & Other Non-Malignant": ["melanocytic nevi", "acne vulgaris"],
    "3. Malignant Skin Tumors": ["squamous cell carcinoma in situ", "basal cell carcinoma"],
}

# Active disease set and group config based on GROUP_MODE + TOP_N
ACTIVE_TOP_DISEASES = TOP_15_DISEASES if TOP_N == 15 else TOP_10_DISEASES
if GROUP_MODE == "4group":
    DISEASE_GROUPS = _DISEASE_GROUPS_4 if TOP_N == 15 else _DISEASE_GROUPS_4_TOP10
else:
    DISEASE_GROUPS = _DISEASE_GROUPS_3 if TOP_N == 15 else _DISEASE_GROUPS_3_TOP10

# Build reverse lookup: disease_name -> group_name
DISEASE_TO_GROUP = {}
for _group, _diseases in DISEASE_GROUPS.items():
    for _disease in _diseases:
        DISEASE_TO_GROUP[norm_disease(_disease)] = _group


def categorize_morphology(disease):
    """
    Map disease label to one of the active groups (Strategy G).
    Must match training code (train_three_stage_hybrid_topk.py).
    """
    return DISEASE_TO_GROUP.get(norm_disease(disease), "Unknown")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(config: InferenceConfig) -> List[Dict]:
    """Load and preprocess data"""
    print("Loading SkinCAP dataset...")

    csv_path = Path(config.CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")

    df = df.dropna(subset=["skincap_file_path", "caption_zh_polish_en", "disease"])
    print(f"  After filtering: {len(df)} valid rows")

    image_base = Path(config.IMAGE_BASE_PATH)
    data_list = []

    for idx, row in df.iterrows():
        img_path = image_base / row["skincap_file_path"]
        if img_path.exists():
            data_list.append({
                "id": int(row.get("id", idx)),
                "image_path": str(img_path),
                "file_name": row["skincap_file_path"],
                "disease": str(row["disease"]).strip().lower(),
                "caption": row["caption_zh_polish_en"],
            })

    print(f"  Found {len(data_list)} valid samples")

    # Filter to top-N diseases FIRST (same order as training script)
    before_filter = len(data_list)
    data_list = [item for item in data_list if norm_disease(item["disease"]) in ACTIVE_TOP_DISEASES]
    print(f"  Strategy G filter: {before_filter} -> {len(data_list)} samples (top-{TOP_N} diseases only)")

    # Fuzzy consolidation AFTER filtering (same order as training — avoids cross-group canonical mapping)
    data_list = fuzzy_consolidate_diseases(data_list, threshold=config.FUZZY_THRESHOLD)

    # Map diseases to groups (same as training code)
    print("  Mapping diseases to groups...")
    for item in data_list:
        item['disease_group'] = categorize_morphology(item['disease'])

    # Verify group distribution
    group_counts = Counter(item['disease_group'] for item in data_list)
    print(f"  Group distribution:")
    for group, count in sorted(group_counts.items()):
        print(f"    {group}: {count} samples")

    return data_list


def get_split_from_info(config: InferenceConfig, full_data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[str]]:
    """Load split info from file and extract matching data"""
    split_path = Path(config.SPLIT_INFO_PATH)

    if not split_path.exists():
        print(f"⚠ Split info not found: {split_path}")
        print(f"  Cannot load training split without split_info file")
        return [], [], GROUP_NAMES

    print(f"Loading split info from: {split_path}")
    with open(split_path, 'r', encoding='utf-8') as f:
        split_info = json.load(f)

    # Extract file paths
    train_paths = split_info.get("train_image_paths", [])
    val_paths = split_info.get("val_image_paths", [])

    # Extract filename numbers
    train_filenames = set()
    for path in train_paths:
        # Get the filename: "SkinCAP\\skincap\\1422.png" -> "1422.png"
        filename = path.split("\\")[-1] if "\\" in path else path.split("/")[-1]
        train_filenames.add(filename)

    val_filenames = set()
    for path in val_paths:
        filename = path.split("\\")[-1] if "\\" in path else path.split("/")[-1]
        val_filenames.add(filename)

    print(f"  Train paths: {len(train_filenames)}")
    print(f"  Val paths: {len(val_filenames)}")

    # Match data by filename
    train_data = []
    val_data = []

    for item in full_data:
        # Extract filename from file_name (could be "skincap\\1422.png" or "skincap/1422.png")
        fname = item["file_name"].split("\\")[-1] if "\\" in item["file_name"] else item["file_name"].split("/")[-1]

        if fname in train_filenames:
            train_data.append(item)
        elif fname in val_filenames:
            val_data.append(item)

    print(f"[OK] Split loaded:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")

    return train_data, val_data, GROUP_NAMES


def normalize_group_name(name: str) -> str:
    """Normalize group name for matching"""
    if not name:
        return ""
    name = name.lower().strip()
    # Remove leading numbers and dots (e.g., "1. Pigmented Lesions" -> "pigmented lesions")
    name = re.sub(r'^[\d\.]+\s*', '', name)
    name = name.replace("-", " ").replace("_", " ")
    return " ".join(name.split())


def match_group_names(pred: str, ground_truth: str) -> bool:
    """Check if prediction matches ground truth group name"""
    pred_norm = normalize_group_name(pred)
    gt_norm = normalize_group_name(ground_truth)

    if pred_norm == gt_norm:
        return True

    pred_words = set(pred_norm.split())
    gt_words = set(gt_norm.split())

    # If ground truth words are subset of prediction
    if gt_words.issubset(pred_words) and len(gt_words) >= 2:
        return True
    # If prediction words are subset of ground truth
    if pred_words.issubset(gt_words) and len(pred_words) >= 2:
        return True

    return False


def create_group_lookup(group_list: List[str]) -> Dict[str, str]:
    """Create normalized group name lookup"""
    lookup = {}
    for group in group_list:
        normalized = normalize_group_name(group)
        lookup[normalized] = group
    return lookup


# ============================================================================
# MODEL & INFERENCE
# ============================================================================

class GroupClassificationInference:
    def __init__(self, model_path: str):
        print(f"\nLoading Group Classification Model (Stage 1)")
        print(f"Model path: {model_path}")

        from unsloth import FastVisionModel

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path, load_in_4bit=True,
        )
        FastVisionModel.for_inference(self.model)
        self.model.eval()

        # Use same prompt as training (first prompt variant) — MUST match GROUP_MODE
        if GROUP_MODE == "4group":
            self.prompt = (
                "Classify this skin condition into one of these 4 medical categories:\n"
                "1. Inflammatory & Autoimmune Diseases\n"
                "2. Benign Tumors, Nevi & Cysts\n"
                "3. Malignant Skin Tumors\n"
                "4. Acne & Follicular Disorders\n\n"
                "Answer with only the category number and name."
            )
        else:
            self.prompt = (
                "Classify this skin condition into one of these 3 medical categories:\n"
                "1. Inflammatory & Autoimmune Diseases\n"
                "2. Benign & Other Non-Malignant\n"
                "3. Malignant Skin Tumors\n\n"
                "Answer with only the category number and name."
            )
        print("[OK] Model loaded!")

    def _load_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def _create_messages(self, image: Image.Image) -> List[Dict]:
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.prompt},
            ],
        }]

    def _extract_group(self, response: str) -> str:
        """Extract group name from model response"""
        # DEBUG: Print raw response to understand format
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1

        # Only print first 5 samples to avoid spam
        if self._debug_count <= 5:
            print(f"\n[DEBUG {self._debug_count}] Raw response: {response[:300]}")

        # Remove <think> tags if present (Qwen3-VL-8B-Thinking model)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = response.strip()

        # Take first line only — prevents garbled multi-line outputs
        response = response.split('\n')[0].strip()

        # Truncate at second number pattern to stop garbled listings
        # e.g. "malignant & 2. benign..." → "malignant"
        truncate_match = re.match(r'^(.+?)\s*\d+\.', response)
        if truncate_match:
            response = truncate_match.group(1).strip()

        response = response.lower()
        extracted = None

        # Try pattern: "group is X" or "classification is X"
        match = re.search(r"(?:group|classification) is\s+(.+?)(?:\.\s*$|$)", response, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()

        # Fall back to entire response (cleaned)
        if not extracted:
            response = re.sub(r"^(the|a|an)\s+", "", response)
            response = re.sub(r"\.\s*$", "", response)
            extracted = response.strip()

        return normalize_group_name(extracted)

    def _match_to_group_list(self, prediction: str, group_list: List[str]) -> Optional[str]:
        """Match prediction to one of the known groups"""
        pred_normalized = normalize_group_name(prediction)
        group_lookup = create_group_lookup(group_list)

        # Exact match
        if pred_normalized in group_lookup:
            return group_lookup[pred_normalized]

        # Fuzzy matching by word overlap
        pred_words = set(pred_normalized.split())
        best_match = None
        best_score = 0

        for norm_group, orig_group in group_lookup.items():
            group_words = set(norm_group.split())
            overlap = len(pred_words & group_words)
            union = len(pred_words | group_words)
            similarity = overlap / union if union > 0 else 0
            score = overlap + similarity

            if score > best_score:
                best_score = score
                best_match = orig_group

        if best_score >= 1:
            return best_match
        return None

    def predict_single(self, image_path: str) -> Dict:
        try:
            image = self._load_image(image_path)
            messages = self._create_messages(image)

            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
                max_length=4096, truncation=False,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=32,  # group name is ~8 tokens; 128 caused looping
                    do_sample=False, use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "assistant" in generated_text.lower():
                response = generated_text.split("assistant")[-1].strip()
            else:
                input_length = inputs["input_ids"].shape[1]
                response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

            predicted_group = self._extract_group(response)
            return {"predicted_group": predicted_group, "status": "success"}

        except Exception as e:
            return {"predicted_group": None, "status": f"error: {str(e)}"}

    def _safe_load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            return self._load_image(image_path)
        except:
            return None

    def predict_batch(self, data: List[Dict], batch_size: int = 8) -> List[Dict]:
        results = []
        total = len(data)

        print(f"\n==> Batch inference: batch_size={batch_size}, total={total}")

        pbar = tqdm(total=total, desc="Inference", dynamic_ncols=True, unit="img",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_data = data[batch_start:batch_end]

            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                loaded_images = list(executor.map(
                    lambda x: self._safe_load_image(x["image_path"]), batch_data
                ))

            batch_messages = []
            valid_indices = []
            batch_valid = []

            for i, (item, image) in enumerate(zip(batch_data, loaded_images)):
                if image is not None:
                    batch_messages.append(self._create_messages(image))
                    valid_indices.append(i)
                    batch_valid.append(True)
                else:
                    batch_valid.append(False)

            if batch_messages:
                try:
                    batch_inputs = self.tokenizer.apply_chat_template(
                        batch_messages, tokenize=True, add_generation_prompt=True,
                        return_tensors="pt", return_dict=True,
                        padding=True, max_length=4096, truncation=True,
                    ).to(self.model.device)

                    with torch.no_grad():
                        outputs = self.model.generate(
                            **batch_inputs, max_new_tokens=32,  # group name is ~8 tokens
                            do_sample=False, use_cache=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    batch_responses = []
                    for i in range(outputs.shape[0]):
                        generated_text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                        if "assistant" in generated_text.lower():
                            response = generated_text.split("assistant")[-1].strip()
                        else:
                            response = generated_text

                        predicted_group = self._extract_group(response)
                        batch_responses.append({
                            "predicted_group": predicted_group,
                            "status": "success"
                        })

                except Exception as e:
                    batch_responses = [{"predicted_group": None, "status": f"error"}
                                      for _ in batch_messages]

            response_idx = 0
            for i, item in enumerate(batch_data):
                if batch_valid[i]:
                    resp = batch_responses[response_idx]
                    response_idx += 1
                else:
                    resp = {"predicted_group": None, "status": "error: failed to load"}

                results.append({
                    "id": item.get("id", ""),
                    "file_name": item.get("file_name", item.get("image_path", "")),
                    "image_path": item["image_path"],
                    "ground_truth": item.get("disease_group", "unknown"),  # Group from training data
                    "predicted": resp["predicted_group"],
                    "status": resp["status"],
                })

            pbar.update(len(batch_data))
            torch.cuda.empty_cache()

        pbar.close()
        return results


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(results: List[Dict], group_list: List[str]) -> Dict:
    """Calculate classification metrics for group predictions"""
    valid_results = [r for r in results if r["status"] == "success" and r["predicted"] is not None]

    if not valid_results:
        return {"error": "No valid predictions", "total_samples": len(results)}

    group_lookup = create_group_lookup(group_list)

    def find_matching_label(pred: str) -> Optional[str]:
        pred_norm = normalize_group_name(pred)
        if pred_norm in group_lookup:
            return group_lookup[pred_norm]
        pred_words = set(pred_norm.split())
        best_match = None
        best_overlap = 0
        for norm_label, orig_label in group_lookup.items():
            label_words = set(norm_label.split())
            if label_words.issubset(pred_words) or pred_words.issubset(label_words):
                overlap = len(pred_words & label_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = orig_label
        return best_match

    y_true = []
    y_pred = []

    for r in valid_results:
        gt = r["ground_truth"]
        pred = r["predicted"]
        matched_pred = find_matching_label(pred)
        y_true.append(gt)
        y_pred.append(matched_pred if matched_pred else pred)

    label_to_idx = {label: idx for idx, label in enumerate(group_list)}
    y_true_idx = [label_to_idx.get(t, -1) for t in y_true]
    y_pred_idx = [label_to_idx.get(p, -1) for p in y_pred]

    valid_pairs = [(t, p) for t, p in zip(y_true_idx, y_pred_idx) if t != -1 and p != -1]

    if not valid_pairs:
        return {"error": "No valid pairs", "total_samples": len(results)}

    y_true_valid = np.array([p[0] for p in valid_pairs])
    y_pred_valid = np.array([p[1] for p in valid_pairs])

    metrics = {
        "total_samples": len(results),
        "successful_predictions": len(valid_results),
        "valid_predictions": len(valid_pairs),
        "num_groups": len(group_list),
        "accuracy": accuracy_score(y_true_valid, y_pred_valid),
        "balanced_accuracy": balanced_accuracy_score(y_true_valid, y_pred_valid),
    }

    for avg in ["macro", "micro", "weighted"]:
        metrics[f"precision_{avg}"] = precision_score(y_true_valid, y_pred_valid, average=avg, zero_division=0)
        metrics[f"recall_{avg}"] = recall_score(y_true_valid, y_pred_valid, average=avg, zero_division=0)
        metrics[f"f1_{avg}"] = f1_score(y_true_valid, y_pred_valid, average=avg, zero_division=0)

    metrics["matthews_corrcoef"] = matthews_corrcoef(y_true_valid, y_pred_valid)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true_valid, y_pred_valid)

    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=list(range(len(group_list))))
    metrics["confusion_matrix"] = cm.tolist()

    mcm = multilabel_confusion_matrix(y_true_valid, y_pred_valid, labels=list(range(len(group_list))))
    per_group_metrics = {}
    for i, group in enumerate(group_list):
        tn, fp, fn, tp = mcm[i].ravel()
        per_group_metrics[group] = {
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
            "support": int(tp + fn),
        }
    metrics["per_group_detailed"] = per_group_metrics

    metrics["overall_score"] = (
        metrics["accuracy"] * 0.3 +
        metrics["balanced_accuracy"] * 0.2 +
        metrics["f1_macro"] * 0.3 +
        (metrics["matthews_corrcoef"] + 1) / 2 * 0.2
    )

    return metrics


def save_results(results: Dict, output_dir: str, split_name: str):
    """Save results to JSON file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / f"results_group_{split_name}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"[OK] Results saved: {filename}")


def print_metrics(metrics: Dict, split_name: str):
    """Print metrics summary"""
    print(f"\n{'='*60}")
    print(f"METRICS - GROUP CLASSIFICATION {split_name.upper()}")
    print(f"{'='*60}")
    print(f"Samples: {metrics.get('total_samples', 'N/A')}")
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
    print(f"F1 (Macro): {metrics.get('f1_macro', 0):.4f}")
    print(f"F1 (Weighted): {metrics.get('f1_weighted', 0):.4f}")
    print(f"Cohen's Kappa: {metrics.get('cohen_kappa', 0):.4f}")
    print(f"Overall Score: {metrics.get('overall_score', 0):.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Group Classification Inference (Stage 1)")
    parser.add_argument("--flow", type=str, default="val",
                        choices=["val", "whole", "both"],
                        help="Inference flow")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    config = InferenceConfig()
    set_seed(config.SEED)

    print(f"{'='*60}")
    print(f"Group Classification Inference (Stage 1)")
    print(f"{'='*60}")

    # Load data
    full_data = load_data(config)

    # Load model
    model = GroupClassificationInference(config.MODEL_PATH_MERGED)

    # NOTE: For group classification, we need to add the disease_group field to each sample
    # This requires loading the training split info to get the group mappings
    # For now, this is a simplified version - you may need to enhance it based on your data structure

    # FLOW 1: Validation set
    if args.flow in ["val", "both"]:
        print(f"\n{'='*60}")
        print("FLOW 1: Validation Set (from training split)")
        print(f"{'='*60}")

        try:
            _, val_data, groups = get_split_from_info(config, full_data)

            if not val_data:
                print("[ERROR] No validation data found!")
            else:
                print(f"Groups ({len(groups)}):")
                for i, grp in enumerate(groups, 1):
                    print(f"  {i}. {grp}")

                val_results = model.predict_batch(val_data, batch_size=args.batch_size)
                val_metrics = calculate_metrics(val_results, groups)

                save_results(val_metrics, config.OUTPUT_DIR, "val")
                save_results({"predictions": val_results}, config.OUTPUT_DIR, "val_predictions")
                print_metrics(val_metrics, "val")

        except Exception as e:
            print(f"[ERROR] Flow 1 error: {e}")
            import traceback
            traceback.print_exc()

    # FLOW 2: Whole dataset
    if args.flow in ["whole", "both"]:
        print(f"\n{'='*60}")
        print("FLOW 2: Whole Dataset")
        print(f"{'='*60}")

        try:
            print(f"Evaluating on all {len(full_data)} samples")

            all_results = model.predict_batch(full_data, batch_size=args.batch_size)
            all_metrics = calculate_metrics(all_results, GROUP_NAMES)

            save_results(all_metrics, config.OUTPUT_DIR, "whole")
            save_results({"predictions": all_results}, config.OUTPUT_DIR, "whole_predictions")
            print_metrics(all_metrics, "whole")

        except Exception as e:
            print(f"[ERROR] Flow 2 error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Complete! Results in: {config.OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
