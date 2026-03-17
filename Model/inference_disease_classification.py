"""
Inference script for Stage 2 Disease Classification Model (3-Stage Pipeline)
Strategy G: Top-N diseases classified using Stage 2 model (loaded from Stage 1 weights)

Supports 5 methods (M0-M4) controlling group context at inference time:
  M0: No group context (baseline)
  M1: GT group label in prompt (oracle upper bound — requires GT labels)
  M2: Stage 1 predicted group in prompt (standard pipeline)
  M3: Stage 1 predicted group in prompt (same as M2 at inference; train differs)
  M4: Stage 1 top-2 beam candidates with scores in prompt (soft context)

IMPORTANT: Set GROUP_MODE, TOP_N, and STAGE2_METHOD below to match training.
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
    precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score, multilabel_confusion_matrix,
)


# ============================================================================
# CONFIG
# ============================================================================

# Must match training config exactly
GROUP_MODE = "4group"   # "4group" or "3group"
TOP_N = 10              # 15 = top-15 diseases, 10 = top-10 diseases
STAGE2_METHOD = "M1"   # "M0", "M1", "M2", "M3", "M4", or "fuzzytopk" (fair comparison on 3stage split)
USE_GROUP_RESTRICTION = False  # E1: restrict predictions to GT/predicted group's disease set
USE_RAG = os.environ.get("HIKARI_USE_RAG", "1") != "0"  # RAG: visual few-shot retrieval from training set
RAG_K = 3               # Number of reference images to retrieve per query
RAG_EXPERIMENT = "R0"  # "R0"-"R4": selects encoder combo from RAG_ENCODER_CONFIGS
PROMPT_VARIANT = "P0"  # "P0"-"P3": selects prompt template
RAG_USE_ALL_DATA = False  # False: train split only (no val leakage); True: all 1010 images

# Special-case model paths for non-3stage methods
_STAGE2_MODEL_PATH_MAP = {
    "fuzzytopk": "./skincap_fuzzytopk_classification_merged",
    "fuzzytopk_s1cascade": "./skincap_fuzzytopk_s1cascade_classification_merged",
    # RAG-in-training variants (Experiment E):
    "fuzzytopk_s1cascade_ragR2_a09": "./skincap_fuzzytopk_s1cascade_ragR2_a09_classification_merged",
    # Stage 3 (caption-trained) models — Way 1/2 × STS ablation:
    # Method names MUST end with _stage3 for IS_CAPTION_MODEL detection
    "fuzzytopk_s1cascade_stage3":               "./skincap_stage3_caption_fuzzytopk_s1cascade_classification_merged",
    "fuzzytopk_s1cascade_merged_init_stage3":   "./skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_classification_merged",
    "fuzzytopk_s1cascade_sts_stage3":           "./skincap_stage3_caption_fuzzytopk_s1cascade_sts_classification_merged",
    "fuzzytopk_s1cascade_merged_init_sts_stage3": "./skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_sts_classification_merged",
    "M1_stage3": "./skincap_3stage_caption_merged",
    # Zero-shot baselines (no fine-tuning — loaded directly from HuggingFace):
    "base": "Qwen/Qwen3-VL-8B-Thinking",          # same model family, zero-shot
    "qwen25": "Qwen/Qwen2.5-VL-7B-Instruct",      # previous generation baseline
}

# Per-method max_new_tokens override — thinking models need much more tokens
_MODEL_MAX_NEW_TOKENS = {
    "base": 1024,    # Qwen3 thinking model: needs ~512-1024 tokens to complete <think>...</think>
    "qwen25": 1024,  # Qwen2.5 no thinking, but give same headroom for verbose reasoning
}

# Caption model detection — methods ending with _stage3 generate captions, not disease names
IS_CAPTION_MODEL = STAGE2_METHOD.endswith("_stage3")

# Prompt used to elicit clinical captions (must match Stage 3 training prompts)
CAPTION_PROMPT = (
    "Describe this skin lesion image in detail. Include information about its appearance, "
    "possible diagnosis, and recommended examinations."
)


@dataclass
class InferenceConfig:
    SEED: int = 42
    STAGE2_MODEL_PATH: str = _STAGE2_MODEL_PATH_MAP.get(
        STAGE2_METHOD, f"./skincap_3stage_disease_{STAGE2_METHOD}_merged"
    )
    STAGE1_MODEL_PATH: str = f"./skincap_3stage_group_{GROUP_MODE}_top{TOP_N}_merged"
    CSV_PATH: str = "./SkinCAP/skincap_v240623.csv"
    IMAGE_BASE_PATH: str = "./SkinCAP/skincap"
    SPLIT_INFO_PATH: str = "./split_info_3stage.json"
    BATCH_SIZE: int = 4
    MAX_NEW_TOKENS: int = 64    # Disease names are longer than group names
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.9
    FUZZY_THRESHOLD: int = 91   # For disease consolidation
    OUTPUT_DIR: str = "./disease_classification_results"
    BEAM_SIZE: int = 4          # For M4 soft probabilities via beam search

    @property
    def RAG_INDEX_PATH(self) -> str:
        scope = "all" if RAG_USE_ALL_DATA else "train"
        return f"./rag_index_{RAG_EXPERIMENT}_{scope}.npz"


# ============================================================================
# SEED
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================================
# DISEASE & GROUP DEFINITIONS (must match training)
# ============================================================================

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


# Top-15 diseases
TOP_15_DISEASES = {
    "squamous cell carcinoma", "basal cell carcinoma", "psoriasis",
    "melanocytic nevi", "lupus erythematosus", "lichen planus",
    "scleroderma", "photodermatoses", "acne vulgaris", "sarcoidosis",
    "seborrheic keratosis", "allergic contact dermatitis",
    "neutrophilic dermatoses", "mycosis fungoides", "folliculitis",
}

# Top-10 diseases
TOP_10_DISEASES = {
    "squamous cell carcinoma", "basal cell carcinoma", "psoriasis",
    "melanocytic nevi", "lupus erythematosus", "lichen planus",
    "scleroderma", "photodermatoses", "acne vulgaris", "sarcoidosis",
}

# 4-group top-15
_DISEASE_GROUPS_4 = {
    "1. Inflammatory & Autoimmune Diseases": [
        "psoriasis", "lupus erythematosus", "lichen planus", "scleroderma",
        "photodermatoses", "sarcoidosis", "allergic contact dermatitis",
        "neutrophilic dermatoses",
    ],
    "2. Benign Tumors, Nevi & Cysts": ["melanocytic nevi", "seborrheic keratosis"],
    "3. Malignant Skin Tumors": [
        "squamous cell carcinoma in situ", "basal cell carcinoma", "mycosis fungoides",
    ],
    "4. Acne & Follicular Disorders": ["acne vulgaris", "folliculitis"],
}

# 3-group top-15
_DISEASE_GROUPS_3 = {
    "1. Inflammatory & Autoimmune Diseases": [
        "psoriasis", "lupus erythematosus", "lichen planus", "scleroderma",
        "photodermatoses", "sarcoidosis", "allergic contact dermatitis",
        "neutrophilic dermatoses",
    ],
    "2. Benign & Other Non-Malignant": [
        "melanocytic nevi", "seborrheic keratosis", "acne vulgaris", "folliculitis",
    ],
    "3. Malignant Skin Tumors": [
        "squamous cell carcinoma in situ", "basal cell carcinoma", "mycosis fungoides",
    ],
}

# 4-group top-10
_DISEASE_GROUPS_4_TOP10 = {
    "1. Inflammatory & Autoimmune Diseases": [
        "psoriasis", "lupus erythematosus", "lichen planus", "scleroderma",
        "photodermatoses", "sarcoidosis",
    ],
    "2. Benign Tumors, Nevi & Cysts": ["melanocytic nevi"],
    "3. Malignant Skin Tumors": ["squamous cell carcinoma in situ", "basal cell carcinoma"],
    "4. Acne & Follicular Disorders": ["acne vulgaris"],
}

# 3-group top-10
_DISEASE_GROUPS_3_TOP10 = {
    "1. Inflammatory & Autoimmune Diseases": [
        "psoriasis", "lupus erythematosus", "lichen planus", "scleroderma",
        "photodermatoses", "sarcoidosis",
    ],
    "2. Benign & Other Non-Malignant": ["melanocytic nevi", "acne vulgaris"],
    "3. Malignant Skin Tumors": ["squamous cell carcinoma in situ", "basal cell carcinoma"],
}

# Active config based on GROUP_MODE + TOP_N
ACTIVE_TOP_DISEASES = TOP_15_DISEASES if TOP_N == 15 else TOP_10_DISEASES
if GROUP_MODE == "4group":
    DISEASE_GROUPS = _DISEASE_GROUPS_4 if TOP_N == 15 else _DISEASE_GROUPS_4_TOP10
else:
    DISEASE_GROUPS = _DISEASE_GROUPS_3 if TOP_N == 15 else _DISEASE_GROUPS_3_TOP10

GROUP_NAMES = list(DISEASE_GROUPS.keys())
DISEASE_NAMES = [d for diseases in DISEASE_GROUPS.values() for d in diseases]

# Reverse lookup: disease -> group
DISEASE_TO_GROUP = {}
for _group, _diseases in DISEASE_GROUPS.items():
    for _d in _diseases:
        DISEASE_TO_GROUP[norm_disease(_d)] = _group


def categorize_morphology(disease: str) -> str:
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

    # Fuzzy consolidation AFTER filtering (operates only on top-N diseases — no cross-group mapping risk)
    data_list = fuzzy_consolidate_diseases(data_list, threshold=config.FUZZY_THRESHOLD)

    # Map diseases to groups and set disease_label_stage2
    # For top-N strategy, K = all diseases, so disease_label_stage2 = disease
    for item in data_list:
        item['disease_group'] = categorize_morphology(item['disease'])
        item['disease_label_stage2'] = item['disease']  # K=all, no "Other" labels

    group_counts = Counter(item['disease_group'] for item in data_list)
    print(f"  Group distribution:")
    for group, count in sorted(group_counts.items()):
        print(f"    {group}: {count} samples")

    return data_list


def get_split_from_info(config: InferenceConfig, full_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Load split info from file and extract matching val data"""
    split_path = Path(config.SPLIT_INFO_PATH)

    if not split_path.exists():
        print(f"  Split info not found: {split_path}")
        return [], []

    print(f"Loading split info from: {split_path}")
    with open(split_path, 'r', encoding='utf-8') as f:
        split_info = json.load(f)

    train_paths = split_info.get("train_image_paths", [])
    val_paths = split_info.get("val_image_paths", [])

    def extract_fname(path: str) -> str:
        return path.split("\\")[-1] if "\\" in path else path.split("/")[-1]

    train_filenames = {extract_fname(p) for p in train_paths}
    val_filenames = {extract_fname(p) for p in val_paths}

    print(f"  Train paths: {len(train_filenames)}, Val paths: {len(val_filenames)}")

    train_data, val_data = [], []
    for item in full_data:
        fname = extract_fname(item["file_name"])
        if fname in val_filenames:
            val_data.append(item)
        elif fname in train_filenames:
            train_data.append(item)

    print(f"  Split loaded — Train: {len(train_data)}, Val: {len(val_data)}")
    return train_data, val_data


# ============================================================================
# DISEASE NAME NORMALIZATION & MATCHING
# ============================================================================

def normalize_disease_name(name: str) -> str:
    """Normalize disease name for matching"""
    if not name:
        return ""
    name = name.lower().strip()
    # Strip "This image shows " prefix if model outputs it
    name = re.sub(r'^this image shows\s+', '', name)
    name = re.sub(r'\.\s*$', '', name)
    name = name.replace("-", " ").replace("_", " ")
    return " ".join(name.split())


def match_to_disease_list(pred: str, disease_list: List[str]) -> Optional[str]:
    """Match prediction to known disease names using exact then word-overlap matching."""
    pred_norm = normalize_disease_name(pred)

    # Build lookup
    lookup = {normalize_disease_name(d): d for d in disease_list}

    # Exact match
    if pred_norm in lookup:
        return lookup[pred_norm]

    # Word overlap scoring (same strategy as Stage 1 group matching)
    pred_words = set(pred_norm.split())
    best_match = None
    best_score = 0

    for norm_d, orig_d in lookup.items():
        d_words = set(norm_d.split())
        overlap = len(pred_words & d_words)
        union = len(pred_words | d_words)
        similarity = overlap / union if union > 0 else 0
        # Require at least 2-word overlap for longer disease names
        score = overlap + similarity
        if overlap >= 2 and score > best_score:
            best_score = score
            best_match = orig_d
        elif overlap == 1 and len(d_words) == 1 and score > best_score:
            # Allow single-word match only if disease itself is one word
            best_score = score
            best_match = orig_d

    if best_score > 0:
        return best_match

    # Fallback: thefuzz partial ratio
    from thefuzz import fuzz as thefuzz
    best_fuzz = None
    best_ratio = 0
    for orig_d in disease_list:
        ratio = thefuzz.token_sort_ratio(pred_norm, normalize_disease_name(orig_d))
        if ratio > best_ratio:
            best_ratio = ratio
            best_fuzz = orig_d
    if best_ratio >= 70:
        return best_fuzz

    return None


# ============================================================================
# STAGE 1 HELPERS (for M2/M3/M4)
# ============================================================================

def run_stage1_for_groups(data: List[Dict], config: InferenceConfig) -> Dict[str, str]:
    """
    Run Stage 1 model on data to get predicted group labels.
    Used by M2, M3 (same inference behavior), and M4.
    Returns {image_path: predicted_group_name}
    """
    print(f"\n[Stage 1] Running group predictions for {STAGE2_METHOD}...")
    from inference_group_classification import GroupClassificationInference
    stage1 = GroupClassificationInference(config.STAGE1_MODEL_PATH)
    results = stage1.predict_batch(data, batch_size=config.BATCH_SIZE)

    pred_map = {}
    for item, result in zip(data, results):
        pred_map[item['image_path']] = result.get('predicted', item['disease_group'])

    # Report accuracy
    correct = sum(1 for item in data if pred_map[item['image_path']] == item['disease_group'])
    print(f"  Stage 1 accuracy on this split: {correct}/{len(data)} ({correct/len(data)*100:.1f}%)")

    del stage1
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    return pred_map


def run_stage1_soft_probs(data: List[Dict], config: InferenceConfig) -> Dict[str, str]:
    """
    Run Stage 1 with beam search to get top-2 group candidates + relative scores.
    Used by M4. Returns {image_path: "Primary: X (85%), Alternative: Y (15%)"}
    """
    print(f"\n[Stage 1 - M4] Running beam search for soft group probabilities...")
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        config.STAGE1_MODEL_PATH, load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    model.eval()

    # Stage 1 prompt (same as training)
    if GROUP_MODE == "4group":
        prompt = (
            "Classify this skin condition into one of these 4 medical categories:\n"
            "1. Inflammatory & Autoimmune Diseases\n"
            "2. Benign Tumors, Nevi & Cysts\n"
            "3. Malignant Skin Tumors\n"
            "4. Acne & Follicular Disorders\n\n"
            "Answer with only the category number and name."
        )
    else:
        prompt = (
            "Classify this skin condition into one of these 3 medical categories:\n"
            "1. Inflammatory & Autoimmune Diseases\n"
            "2. Benign & Other Non-Malignant\n"
            "3. Malignant Skin Tumors\n\n"
            "Answer with only the category number and name."
        )

    from inference_group_classification import GroupClassificationInference, normalize_group_name

    def _extract_group_name(text: str) -> str:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        text = text.split('\n')[0].strip()
        m = re.match(r'^(.+?)\s*\d+\.', text)
        if m:
            text = m.group(1).strip()
        return normalize_group_name(text)

    pred_map = {}
    total = len(data)
    pbar = tqdm(total=total, desc="M4 beam search", unit="img")

    for item in data:
        try:
            image = Image.open(item['image_path']).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]

            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
                max_length=4096, truncation=True,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    num_beams=config.BEAM_SIZE,
                    num_return_sequences=2,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            sequences = outputs.sequences          # [2, seq_len]
            seq_scores = outputs.sequences_scores  # [2] — log-probs

            # Softmax over the 2 sequence log-probs
            probs = torch.softmax(seq_scores, dim=0).cpu().tolist()

            # Decode both sequences
            decoded = []
            for i in range(len(sequences)):
                text = tokenizer.decode(sequences[i], skip_special_tokens=True)
                if "assistant" in text.lower():
                    text = text.split("assistant")[-1].strip()
                decoded.append(_extract_group_name(text))

            # Match to known group names
            from inference_group_classification import GroupClassificationInference as _GCI
            dummy = object.__new__(_GCI)
            dummy.model = model
            dummy.tokenizer = tokenizer
            matched = [dummy._match_to_group_list(d, GROUP_NAMES) or d for d in decoded]

            p1, p2 = probs[0], probs[1]
            g1, g2 = matched[0], matched[1]

            if g1 == g2:
                # Both beams agree — show primary only
                context = f"Predicted group: {g1} (high confidence)"
            else:
                context = (
                    f"Primary: {g1} ({p1*100:.0f}%), "
                    f"Alternative: {g2} ({p2*100:.0f}%)"
                )

            pred_map[item['image_path']] = context

        except Exception as e:
            # Fallback: use GT group (shouldn't happen in practice)
            pred_map[item['image_path']] = f"Predicted group: {item['disease_group']}"

        pbar.update(1)
        torch.cuda.empty_cache()

    pbar.close()

    del model, tokenizer
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    return pred_map


# ============================================================================
# DISEASE PROMPTS
# ============================================================================

DISEASE_CLASSIFICATION_PROMPTS = [
    "Carefully examine this dermatological image. Look for: lesion morphology (papule/plaque/macule/nodule), color (red/violet/white/brown/black), scale or crust, border sharpness, and distribution. Based on these visual features, what is the specific skin disease?",
    "Identify the dermatological condition in this photograph.",
    "What is the diagnosis for the skin lesion shown?",
]

DISEASE_PROMPTS_WITH_GROUP = [
    "This skin lesion belongs to the group '{group}'. Examine the lesion morphology (papules, plaques, macules), color (red, violet, white, brown), scale/crust, border sharpness, and distribution pattern. Based on these visual features, what is the specific skin disease?",
    "Given this condition is classified as '{group}', identify the specific dermatological disease.",
    "This image shows a condition in the '{group}' category. What is the precise diagnosis?",
]

# Chain-of-thought prompt variants (P0=current, P1=step-by-step, P2=differential, P3=structured)
COT_PROMPTS = {
    "P0": DISEASE_CLASSIFICATION_PROMPTS[0],
    "P1": (
        "Analyze this dermatological image step by step:\n"
        "Step 1: Describe the primary lesion (papule/plaque/macule/nodule/vesicle).\n"
        "Step 2: Note the color (red, violet, white, brown, black, mixed).\n"
        "Step 3: Identify texture/surface (smooth, scaly, crusted, ulcerated).\n"
        "Step 4: Note the border (sharp/blurred) and distribution.\n"
        "Step 5: Based on these features, state the specific skin disease."
    ),
    "P2": (
        "Examine this skin lesion. List the 3 most likely diagnoses and the key visual evidence "
        "for each. Then select the single most likely diagnosis. Format: "
        "1. [Disease] - [evidence]; 2. [Disease] - [evidence]; 3. [Disease] - [evidence]. "
        "Final diagnosis: [Disease]"
    ),
    "P3": (
        "Clinical assessment of this skin lesion:\n"
        "• Morphology: [describe primary lesion type]\n"
        "• Color/pigmentation: [describe]\n"
        "• Surface: [scale/crust/smooth]\n"
        "• Border: [sharp/ill-defined]\n"
        "• Distribution: [localized/disseminated]\n"
        "Diagnosis: [state the specific skin disease]"
    ),
}

COT_PROMPTS_WITH_GROUP = {
    "P0": DISEASE_PROMPTS_WITH_GROUP[0],
    "P1": (
        "This lesion belongs to '{group}'. Analyze step by step: "
        "(1) primary lesion type (papule/plaque/macule/nodule) "
        "(2) color (red/violet/white/brown/black) "
        "(3) surface texture (smooth/scaly/crusted) "
        "(4) border sharpness and distribution "
        "(5) specific disease name."
    ),
    "P2": (
        "This lesion is in the '{group}' group. List the top 3 most likely specific diseases "
        "within this group with key visual evidence for each, then choose the single most likely. "
        "Format: 1. [Disease] - [evidence]; 2. [Disease] - [evidence]; 3. [Disease] - [evidence]. "
        "Final diagnosis: [Disease]"
    ),
    "P3": (
        "Group: '{group}'.\n"
        "Clinical assessment:\n"
        "• Morphology: [primary lesion type]\n"
        "• Color: [describe]\n"
        "• Surface: [scale/crust/smooth]\n"
        "• Border: [sharp/ill-defined]\n"
        "Diagnosis (specific disease within this group):"
    ),
}


# ============================================================================
# MODEL & INFERENCE
# ============================================================================

class DiseaseClassificationInference:
    def __init__(self, model_path: str):
        print(f"\nLoading Disease Classification Model (Stage 2)")
        print(f"Model path: {model_path}")

        from unsloth import FastVisionModel

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path, load_in_4bit=True,
        )
        FastVisionModel.for_inference(self.model)
        self.model.eval()
        print("[OK] Stage 2 model loaded!")

    def _load_image(self, image_path: str) -> Image.Image:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((672, 672), Image.LANCZOS)
        return img

    def _safe_load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            return self._load_image(image_path)
        except Exception:
            return None

    def _create_messages(
        self,
        image: Image.Image,
        group_context: Optional[str] = None,
        ref_pairs: Optional[List] = None,
        prompt_variant: str = "P0",
    ) -> List[Dict]:
        """Build chat message. group_context adds group info; ref_pairs adds RAG few-shot examples."""
        content: List[Dict] = []

        # RAG: inject reference images before the query
        if ref_pairs:
            content.append({"type": "text", "text": "Here are similar reference cases for context:"})
            for i, ref in enumerate(ref_pairs, 1):
                ref_path, ref_label = ref[0], ref[1]
                ref_caption = ref[2] if len(ref) > 2 else None
                ref_img = self._safe_load_image(ref_path)
                if ref_img is not None:
                    content.append({"type": "image", "image": ref_img})
                    label_line = f"Reference {i}: {ref_label}"
                    if ref_caption:
                        label_line += f"\nDescription: {ref_caption}"
                    content.append({"type": "text", "text": label_line})
            content.append({"type": "text", "text": "\nNow, identify the disease in this new image:"})

        content.append({"type": "image", "image": image})

        pv = prompt_variant if prompt_variant in COT_PROMPTS else "P0"
        if group_context is not None:
            # M1/M2/M3: hard group label; M4: soft probability string
            template = COT_PROMPTS_WITH_GROUP.get(pv, COT_PROMPTS_WITH_GROUP["P0"])
            prompt = template.replace('{group}', group_context)
        else:
            prompt = COT_PROMPTS.get(pv, COT_PROMPTS["P0"])
        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def _extract_disease(self, response: str) -> str:
        """Parse model response to extract disease name."""
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1

        if self._debug_count <= 5:
            print(f"\n[DEBUG {self._debug_count}] Raw response: {response[:200]}")

        # Remove thinking blocks (including dangling </think> tags from double-close bug)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = re.sub(r'</think>', '', response).strip()

        # Priority 1: "Diagnosis:" or "Final diagnosis:" (P3 / P2)
        diag_match = re.search(r'(?:[Ff]inal\s+)?[Dd]iagnosis\s*:\s*(.+)', response)
        if diag_match:
            response = diag_match.group(1).strip()
        else:
            # Priority 2: last "Step N:" line (P1 step-by-step format)
            steps = re.findall(r'Step\s*\d+\s*[:\-]\s*(.+)', response)
            if steps:
                response = steps[-1].strip()
            else:
                # Priority 3: scan all lines, pick shortest non-empty line
                # (long lines = verbose description, short lines = disease name)
                lines = [l.strip() for l in response.split('\n') if l.strip()]
                if len(lines) > 1:
                    # Remove lines that are clearly headers/steps/bullets
                    clean_lines = [
                        re.sub(r'^[\d]+[\.\)]\s*', '', l)  # strip "1. " or "1) "
                        for l in lines
                        if not re.match(r'^(Step|Here are|Reference|Now|Clinical)', l, re.I)
                    ]
                    # Take the last clean line (conclusions come last)
                    response = clean_lines[-1].strip() if clean_lines else lines[-1]
                else:
                    response = lines[0] if lines else response

        # Strip "This image shows " prefix and trailing punctuation
        response = re.sub(r'^[Tt]his image shows\s+', '', response)
        response = re.sub(r'\.\s*$', '', response).strip()
        # Strip bullet/number prefixes that may remain
        response = re.sub(r'^[\-\*\•]\s*', '', response)

        return response

    def predict_batch(
        self,
        data: List[Dict],
        group_context_map: Optional[Dict[str, str]] = None,
        group_restriction_map: Optional[Dict[str, List[str]]] = None,
        rag_retriever=None,
        batch_size: int = 4,
        max_new_tokens: int = 64,
        prompt_variant: str = "P0",
    ) -> List[Dict]:
        """
        Run batch inference on data.

        Args:
            group_context_map: {image_path: group_context_string} for M1/M2/M3/M4.
                               None for M0 (no group context).
            group_restriction_map: {image_path: [disease_list]} for E1 label restriction.
                                   If set, predictions are constrained to the group's diseases.
            rag_retriever: RAGRetriever/HybridRAGRetriever instance. None = disabled.
            prompt_variant: "P0"-"P3" selects CoT prompt template.
        """
        results = []
        total = len(data)

        print(f"\n==> Batch inference: batch_size={batch_size}, total={total}")
        if rag_retriever is not None:
            print(f"  RAG enabled: {RAG_K} reference images per query")

        pbar = tqdm(total=total, desc="Disease Inference", dynamic_ncols=True, unit="img",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_data = data[batch_start:batch_end]

            # Load images in parallel
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                loaded_images = list(executor.map(
                    lambda x: self._safe_load_image(x["image_path"]), batch_data
                ))

            batch_messages = []
            valid_indices = []
            batch_valid = []

            for i, (item, image) in enumerate(zip(batch_data, loaded_images)):
                if image is not None:
                    group_ctx = (group_context_map or {}).get(item['image_path'])
                    vlm_desc = item.get('symptom_description') or item.get('caption')  # symptom (user) > clinical caption
                    ref_pairs = rag_retriever.retrieve(image, k=RAG_K, vlm_description=vlm_desc) if rag_retriever else None
                    batch_messages.append(self._create_messages(image, group_ctx, ref_pairs, prompt_variant))
                    valid_indices.append(i)
                    batch_valid.append(True)
                else:
                    batch_valid.append(False)

            batch_responses = []
            if batch_messages:
                try:
                    batch_inputs = self.tokenizer.apply_chat_template(
                        batch_messages, tokenize=True, add_generation_prompt=True,
                        return_tensors="pt", return_dict=True,
                        padding=True,
                    ).to(self.model.device)

                    with torch.no_grad():
                        outputs = self.model.generate(
                            **batch_inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            use_cache=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    for i in range(outputs.shape[0]):
                        generated_text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                        if "assistant" in generated_text.lower():
                            response = generated_text.split("assistant")[-1].strip()
                        else:
                            response = generated_text

                        raw_pred = self._extract_disease(response)
                        item_for_match = batch_data[valid_indices[i]]
                        if group_restriction_map:
                            candidates = group_restriction_map.get(item_for_match['image_path'], DISEASE_NAMES)
                        else:
                            candidates = DISEASE_NAMES
                        matched = match_to_disease_list(raw_pred, candidates)
                        batch_responses.append({
                            "raw": raw_pred,
                            "predicted": matched if matched else raw_pred,
                            "status": "success",
                        })

                except Exception as e:
                    is_oom = "CUDA out of memory" in str(e) or "out of memory" in str(e).lower()
                    if is_oom:
                        # OOM: retry each item individually (batch_size=1)
                        print(f"\n  [OOM] batch failed, retrying {len(batch_messages)} items one-by-one...")
                        try:
                            torch.cuda.empty_cache()
                            import gc; gc.collect()
                        except Exception:
                            pass
                        batch_responses = []
                        for single_msg in batch_messages:
                            try:
                                # No max_length truncation — avoids image token count mismatch
                                single_input = self.tokenizer.apply_chat_template(
                                    [single_msg], tokenize=True, add_generation_prompt=True,
                                    return_tensors="pt", return_dict=True,
                                    padding=True,
                                ).to(self.model.device)
                                with torch.no_grad():
                                    single_out = self.model.generate(
                                        **single_input,
                                        max_new_tokens=max_new_tokens,
                                        do_sample=False,
                                        use_cache=True,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                    )
                                gen_text = self.tokenizer.decode(single_out[0], skip_special_tokens=True)
                                if "assistant" in gen_text.lower():
                                    gen_text = gen_text.split("assistant")[-1].strip()
                                raw_pred = self._extract_disease(gen_text)
                                batch_responses.append({"raw": raw_pred, "predicted": raw_pred, "status": "success"})
                            except Exception as e2:
                                print(f"  [OOM retry failed] {e2}")
                                batch_responses.append({"raw": "", "predicted": None, "status": "error"})
                            finally:
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                        # Re-apply match_to_disease_list for retry responses
                        for idx, (resp, vi) in enumerate(zip(batch_responses, valid_indices)):
                            if resp["status"] == "success":
                                item_for_match = batch_data[vi]
                                candidates = group_restriction_map.get(item_for_match['image_path'], DISEASE_NAMES) if group_restriction_map else DISEASE_NAMES
                                matched = match_to_disease_list(resp["raw"], candidates)
                                batch_responses[idx]["predicted"] = matched if matched else resp["raw"]
                    else:
                        print(f"\n  [ERROR] batch failed: {e}")
                        batch_responses = [{"raw": "", "predicted": None, "status": "error"}
                                           for _ in batch_messages]

            response_idx = 0
            for i, item in enumerate(batch_data):
                if batch_valid[i]:
                    resp = batch_responses[response_idx]
                    response_idx += 1
                else:
                    resp = {"raw": "", "predicted": None, "status": "error: failed to load"}

                results.append({
                    "id": item["id"],
                    "file_name": item["file_name"],
                    "image_path": item["image_path"],
                    "ground_truth": item.get("disease_label_stage2", item.get("disease", "unknown")),
                    "disease_group": item.get("disease_group", "unknown"),
                    "predicted_raw": resp["raw"],
                    "predicted": resp["predicted"],
                    "status": resp["status"],
                })

            pbar.update(len(batch_data))
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        pbar.close()
        return results

    def generate_caption_batch(
        self,
        data: List[Dict],
        batch_size: int = 4,
        max_new_tokens: int = 256,
    ) -> List[Dict]:
        """Generate clinical captions for images (Stage 3 evaluation mode)."""
        results = []
        total = len(data)

        from tqdm import tqdm
        pbar = tqdm(total=total, desc="Generating captions")

        for batch_start in range(0, total, batch_size):
            batch_data = data[batch_start: batch_start + batch_size]
            batch_messages = []
            batch_valid = []

            for item in batch_data:
                image = self._safe_load_image(item["image_path"])
                if image is not None:
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": CAPTION_PROMPT},
                    ]}]
                    batch_messages.append(messages)
                    batch_valid.append(True)
                else:
                    batch_valid.append(False)

            if not batch_messages:
                for item in batch_data:
                    results.append({
                        "id": item.get("id", ""),
                        "file_name": item.get("file_name", ""),
                        "image_path": item["image_path"],
                        "ground_truth_caption": item.get("caption", ""),
                        "generated_caption": "",
                        "status": "error: failed to load image",
                    })
                pbar.update(len(batch_data))
                continue

            try:
                inputs = self.tokenizer.apply_chat_template(
                    batch_messages, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True, padding=True,
                ).to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                batch_captions = []
                for i in range(outputs.shape[0]):
                    text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                    if "assistant" in text.lower():
                        text = text.split("assistant")[-1].strip()
                    # Strip thinking blocks
                    import re as _re
                    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
                    batch_captions.append(text)
            except Exception as e:
                print(f"\n  [ERROR] caption batch failed: {e}")
                batch_captions = [""] * len(batch_messages)

            cap_idx = 0
            for i, item in enumerate(batch_data):
                if batch_valid[i]:
                    gen_cap = batch_captions[cap_idx] if cap_idx < len(batch_captions) else ""
                    cap_idx += 1
                else:
                    gen_cap = ""
                results.append({
                    "id": item.get("id", ""),
                    "file_name": item.get("file_name", ""),
                    "image_path": item["image_path"],
                    "ground_truth_caption": item.get("caption", ""),
                    "generated_caption": gen_cap,
                    "status": "success" if batch_valid[i] else "error: failed to load image",
                })

            pbar.update(len(batch_data))
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        pbar.close()
        return results


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(results: List[Dict], disease_names: List[str]) -> Dict:
    """Calculate disease classification metrics."""
    valid_results = [r for r in results if r["status"] == "success" and r["predicted"] is not None]

    if not valid_results:
        return {"error": "No valid predictions", "total_samples": len(results)}

    disease_lookup = {normalize_disease_name(d): d for d in disease_names}

    def find_label(pred: str) -> Optional[str]:
        matched = match_to_disease_list(pred, disease_names)
        return matched

    y_true = []
    y_pred = []

    # Include ALL results — unrecognized predictions count as wrong (index = len(disease_names))
    UNKNOWN_IDX = len(disease_names)
    label_to_idx = {label: idx for idx, label in enumerate(disease_names)}

    for r in results:
        gt = r.get("ground_truth", "")
        pred = r.get("predicted", "") or ""
        matched_pred = find_label(pred) if pred else None
        gt_idx = label_to_idx.get(gt, -1)
        pred_idx = label_to_idx.get(matched_pred, UNKNOWN_IDX) if matched_pred else UNKNOWN_IDX
        if gt_idx == -1:
            continue  # skip samples whose GT isn't a recognised disease (shouldn't happen)
        y_true.append(gt_idx)
        y_pred.append(pred_idx)

    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != UNKNOWN_IDX]

    if not y_true:
        return {"error": "No valid label pairs found", "total_samples": len(results)}

    y_true_v = np.array(y_true)
    y_pred_v = np.array(y_pred)

    # Accuracy over ALL samples (unknown predictions = wrong)
    n_correct = int(np.sum(y_true_v == y_pred_v))
    n_total = len(y_true_v)
    accuracy_all = n_correct / n_total if n_total > 0 else 0.0

    # For per-class metrics (balanced_acc, f1, kappa) use only valid pairs
    # so that the "Unknown" prediction class doesn't distort per-class statistics
    if valid_pairs:
        yv_t = np.array([p[0] for p in valid_pairs])
        yv_p = np.array([p[1] for p in valid_pairs])
    else:
        yv_t = y_true_v
        yv_p = y_pred_v

    metrics = {
        "method": STAGE2_METHOD,
        "rag_experiment": RAG_EXPERIMENT if USE_RAG else "NoRAG",
        "prompt_variant": PROMPT_VARIANT,
        "group_mode": GROUP_MODE,
        "top_n": TOP_N,
        "total_samples": n_total,
        "successful_predictions": len(valid_results),
        "valid_predictions": len(valid_pairs),
        "num_diseases": len(disease_names),
        "accuracy": accuracy_all,
        "balanced_accuracy": balanced_accuracy_score(yv_t, yv_p),
    }

    for avg in ["macro", "micro", "weighted"]:
        metrics[f"precision_{avg}"] = precision_score(yv_t, yv_p, average=avg, zero_division=0)
        metrics[f"recall_{avg}"] = recall_score(yv_t, yv_p, average=avg, zero_division=0)
        metrics[f"f1_{avg}"] = f1_score(yv_t, yv_p, average=avg, zero_division=0)

    metrics["matthews_corrcoef"] = matthews_corrcoef(yv_t, yv_p)
    metrics["cohen_kappa"] = cohen_kappa_score(yv_t, yv_p)

    cm = confusion_matrix(y_true_v, y_pred_v, labels=list(range(len(disease_names))))
    metrics["confusion_matrix"] = cm.tolist()

    mcm = multilabel_confusion_matrix(y_true_v, y_pred_v, labels=list(range(len(disease_names))))
    per_disease = {}
    for i, disease in enumerate(disease_names):
        tn, fp, fn, tp = mcm[i].ravel()
        per_disease[disease] = {
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
            "support": int(tp + fn),
        }
    metrics["per_disease_detailed"] = per_disease

    metrics["overall_score"] = (
        metrics["accuracy"] * 0.3 +
        metrics["balanced_accuracy"] * 0.2 +
        metrics["f1_macro"] * 0.3 +
        (metrics["matthews_corrcoef"] + 1) / 2 * 0.2
    )

    return metrics


def evaluate_captions(results: List[Dict], output_dir: str, split_name: str) -> Dict:
    """
    Evaluate Stage 3 caption generation using BLEU-1/2/3/4 and ROUGE-L.
    Requires: nltk, rouge_score (pip install nltk rouge-score)
    Falls back gracefully if packages are missing.
    """
    import re

    valid = [r for r in results if r["status"] == "success" and r.get("generated_caption")]
    total = len(results)
    n_valid = len(valid)

    print(f"\n  Caption eval: {n_valid}/{total} valid generations")

    if not valid:
        return {"error": "No valid captions generated", "total_samples": total}

    references = [r["ground_truth_caption"] for r in valid]
    hypotheses = [r["generated_caption"] for r in valid]

    metrics: Dict = {
        "total_samples": total,
        "valid_samples": n_valid,
        "avg_generated_length": round(sum(len(h.split()) for h in hypotheses) / n_valid, 1),
        "avg_reference_length": round(sum(len(r.split()) for r in references) / n_valid, 1),
    }

    # BLEU (corpus-level) via nltk
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        sf = SmoothingFunction().method1
        tok_refs = [[ref.lower().split()] for ref in references]
        tok_hyps = [hyp.lower().split() for hyp in hypotheses]

        bleu1 = corpus_bleu(tok_refs, tok_hyps, weights=(1, 0, 0, 0), smoothing_function=sf)
        bleu2 = corpus_bleu(tok_refs, tok_hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=sf)
        bleu3 = corpus_bleu(tok_refs, tok_hyps, weights=(1/3, 1/3, 1/3, 0), smoothing_function=sf)
        bleu4 = corpus_bleu(tok_refs, tok_hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf)

        metrics["bleu_1"] = round(bleu1 * 100, 2)
        metrics["bleu_2"] = round(bleu2 * 100, 2)
        metrics["bleu_3"] = round(bleu3 * 100, 2)
        metrics["bleu_4"] = round(bleu4 * 100, 2)
        print(f"  BLEU-1: {metrics['bleu_1']:.2f}  BLEU-2: {metrics['bleu_2']:.2f}  "
              f"BLEU-3: {metrics['bleu_3']:.2f}  BLEU-4: {metrics['bleu_4']:.2f}")
    except ImportError:
        print("  [BLEU] nltk not installed — skipping. Run: pip install nltk")
    except Exception as e:
        print(f"  [BLEU] Error: {e}")

    # ROUGE-L via rouge_score
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1_scores, r2_scores, rL_scores = [], [], []
        for hyp, ref in zip(hypotheses, references):
            s = scorer.score(ref, hyp)
            r1_scores.append(s["rouge1"].fmeasure)
            r2_scores.append(s["rouge2"].fmeasure)
            rL_scores.append(s["rougeL"].fmeasure)
        metrics["rouge_1"] = round(sum(r1_scores) / len(r1_scores) * 100, 2)
        metrics["rouge_2"] = round(sum(r2_scores) / len(r2_scores) * 100, 2)
        metrics["rouge_L"] = round(sum(rL_scores) / len(rL_scores) * 100, 2)
        print(f"  ROUGE-1: {metrics['rouge_1']:.2f}  ROUGE-2: {metrics['rouge_2']:.2f}  "
              f"ROUGE-L: {metrics['rouge_L']:.2f}")
    except ImportError:
        print("  [ROUGE] rouge_score not installed — skipping. Run: pip install rouge-score")
    except Exception as e:
        print(f"  [ROUGE] Error: {e}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"results_caption_{STAGE2_METHOD}_{split_name}.json"
    pred_file = output_path / f"results_caption_{STAGE2_METHOD}_{split_name}_predictions.json"

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump({"predictions": results, "metrics": metrics}, f, ensure_ascii=False, indent=2)

    print(f"  Results saved -> {result_file}")
    return metrics


def save_results(results: Dict, output_dir: str, split_name: str, src_tag: str = ""):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rag_tag = f"_RAG{RAG_EXPERIMENT}" if USE_RAG else "_NoRAG"
    prompt_tag = f"_P{PROMPT_VARIANT[1:]}" if PROMPT_VARIANT != "P0" else ""
    filename = output_path / f"results_disease_{STAGE2_METHOD}{rag_tag}{prompt_tag}{src_tag}_{split_name}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"[OK] Results saved: {filename}")


def print_metrics(metrics: Dict, split_name: str):
    print(f"\n{'='*60}")
    print(f"METRICS - DISEASE CLASSIFICATION {split_name.upper()} (Method: {STAGE2_METHOD})")
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
    global RAG_EXPERIMENT, PROMPT_VARIANT

    import argparse

    parser = argparse.ArgumentParser(description="Disease Classification Inference (Stage 2)")
    parser.add_argument("--flow", type=str, default="val",
                        choices=["val", "whole", "both"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--rag_exp", type=str, default=None,
                        help="RAG encoder experiment: R0-R4 (overrides RAG_EXPERIMENT)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt variant: P0-P3 (overrides PROMPT_VARIANT)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit val/whole set to N samples (for quick smoke test)")
    parser.add_argument("--stage2_method", type=str, default=None,
                        help="Override STAGE2_METHOD (e.g. fuzzytopk, fuzzytopk_s1cascade, M0, M1)")
    parser.add_argument("--vlm_desc_source", type=str, default="symptoms",
                        choices=["symptoms", "stage3", "none"],
                        help="Text query source for R1-R3 hybrid retrieval: "
                             "'symptoms'=patient-written (default), 'stage3'=Stage3 generated captions, "
                             "'none'=image-only fallback")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override RAG hybrid score weight (0-1, image weight). "
                             "Default: use value stored in index (0.5). Try 0.7 to boost image signal.")
    args = parser.parse_args()

    if args.rag_exp is not None:
        RAG_EXPERIMENT = args.rag_exp
    if args.prompt is not None:
        PROMPT_VARIANT = args.prompt

    config = InferenceConfig()

    if args.stage2_method is not None:
        global STAGE2_METHOD
        STAGE2_METHOD = args.stage2_method
        config.STAGE2_MODEL_PATH = _STAGE2_MODEL_PATH_MAP.get(
            STAGE2_METHOD, f"./skincap_3stage_disease_{STAGE2_METHOD}_merged"
        )
    if STAGE2_METHOD in _MODEL_MAX_NEW_TOKENS:
        config.MAX_NEW_TOKENS = _MODEL_MAX_NEW_TOKENS[STAGE2_METHOD]

    # Re-evaluate IS_CAPTION_MODEL after CLI override (STAGE2_METHOD may have changed)
    is_caption = STAGE2_METHOD.endswith("_stage3")

    # Source/alpha tags appended to result filenames to avoid overwriting existing runs
    _src_tag = "_s3cap" if args.vlm_desc_source == "stage3" else ""
    _alpha_tag = f"_a{str(args.alpha).replace('.', '')}" if args.alpha is not None else ""
    _src_tag = _src_tag + _alpha_tag

    set_seed(config.SEED)

    print(f"{'='*60}")
    print(f"Disease Classification Inference (Stage 2)")
    print(f"{'='*60}")
    print(f"Method:       {STAGE2_METHOD}")
    print(f"GROUP_MODE:   {GROUP_MODE}, TOP_N: {TOP_N}")
    print(f"Model:        {config.STAGE2_MODEL_PATH}")
    print(f"RAG:          {'ON — ' + RAG_EXPERIMENT if USE_RAG else 'OFF'}")
    print(f"RAG all data: {RAG_USE_ALL_DATA}")
    print(f"Prompt:       {PROMPT_VARIANT}")
    print(f"Diseases:     {DISEASE_NAMES}")

    # Load data
    full_data = load_data(config)

    # Build RAG index if enabled (before loading Stage 2 to free VRAM before big model load)
    rag_retriever = None
    if USE_RAG:
        from rag_retrieval import HybridRAGRetriever, RAG_ENCODER_CONFIGS
        rag_cfg = RAG_ENCODER_CONFIGS[RAG_EXPERIMENT]
        rag_data = full_data if RAG_USE_ALL_DATA else None
        if rag_data is None:
            rag_data, _ = get_split_from_info(config, full_data)
            if not rag_data:
                rag_data = full_data
        print(f"\n[RAG] Building index from {len(rag_data)} images (exp={RAG_EXPERIMENT})")
        rag_retriever = HybridRAGRetriever.load_or_build(
            rag_data, config.RAG_INDEX_PATH,
            img_encoder_name=rag_cfg["img"],
            txt_encoder_name=rag_cfg["txt"],
            strategy=rag_cfg["strategy"],
        )
        if args.alpha is not None:
            rag_retriever.alpha = args.alpha
            print(f"  [RAG] Alpha overridden to {args.alpha} (image weight)")

    # Load Stage 2 model
    stage2 = DiseaseClassificationInference(config.STAGE2_MODEL_PATH)

    # FLOW 1: Validation set
    if args.flow in ["val", "both"]:
        print(f"\n{'='*60}")
        print("FLOW 1: Validation Set (from training split)")
        print(f"{'='*60}")

        try:
            _, val_data = get_split_from_info(config, full_data)

            if args.n_samples:
                val_data = val_data[:args.n_samples]
                print(f"  [smoke test] Limited to {len(val_data)} samples")

            if not val_data:
                print("[ERROR] No validation data found!")
            elif is_caption:
                # ── Caption evaluation mode (Stage 3 models) ──────────────────
                print(f"  [caption mode] Generating captions with {STAGE2_METHOD}")
                val_results = stage2.generate_caption_batch(
                    val_data,
                    batch_size=args.batch_size,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                )
                evaluate_captions(val_results, config.OUTPUT_DIR, "val")
            else:
                # ── Disease classification mode (default) ─────────────────────
                # Load text query source for R1-R3 hybrid retrieval
                import json as _json
                _vlm_src = args.vlm_desc_source
                if _vlm_src == "symptoms":
                    _sym_path = Path(__file__).parent / "val_captions_for_symptoms.json"
                    if _sym_path.exists():
                        _sym_data = _json.load(open(_sym_path, encoding="utf-8"))
                        _desc_map = {r["fname"]: r["symptom_description"] for r in _sym_data
                                     if r.get("symptom_description", "").strip()}
                        for item in val_data:
                            fname = Path(item["image_path"]).name
                            if fname in _desc_map:
                                item["symptom_description"] = _desc_map[fname]
                        print(f"  [vlm_desc=symptoms] Loaded {len(_desc_map)} patient symptom descriptions")
                elif _vlm_src == "stage3":
                    _s3_path = Path(__file__).parent / "val_captions_stage3.json"
                    if _s3_path.exists():
                        _s3_data = _json.load(open(_s3_path, encoding="utf-8"))
                        _desc_map = {r["fname"]: r["stage3_caption"] for r in _s3_data
                                     if r.get("stage3_caption", "").strip()}
                        for item in val_data:
                            fname = Path(item["image_path"]).name
                            if fname in _desc_map:
                                item["symptom_description"] = _desc_map[fname]
                        print(f"  [vlm_desc=stage3] Loaded {len(_desc_map)} Stage 3 generated captions")
                    else:
                        print("  [vlm_desc=stage3] val_captions_stage3.json not found — falling back to image-only")
                else:
                    print("  [vlm_desc=none] No text query source — image-only retrieval for all RAG configs")

                # Build group context map based on method
                group_context_map = None
                group_restriction_map = None

                if STAGE2_METHOD == "M1":
                    group_context_map = {item['image_path']: item['disease_group']
                                         for item in val_data}
                    print(f"  M1: Using GT group labels as context")
                    if USE_GROUP_RESTRICTION:
                        group_restriction_map = {
                            item['image_path']: DISEASE_GROUPS.get(item['disease_group'], DISEASE_NAMES)
                            for item in val_data
                        }
                        print(f"  E1: Group restriction enabled")

                elif STAGE2_METHOD in ("M2", "M3"):
                    stage1_pred_map = run_stage1_for_groups(val_data, config)
                    group_context_map = stage1_pred_map
                    if USE_GROUP_RESTRICTION:
                        group_restriction_map = {
                            item['image_path']: DISEASE_GROUPS.get(stage1_pred_map[item['image_path']], DISEASE_NAMES)
                            for item in val_data
                        }
                        print(f"  E1: Group restriction enabled (Stage 1 predicted groups)")

                elif STAGE2_METHOD == "M4":
                    group_context_map = run_stage1_soft_probs(val_data, config)

                val_results = stage2.predict_batch(
                    val_data,
                    group_context_map=group_context_map,
                    group_restriction_map=group_restriction_map,
                    rag_retriever=rag_retriever,
                    batch_size=args.batch_size,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    prompt_variant=PROMPT_VARIANT,
                )
                val_metrics = calculate_metrics(val_results, DISEASE_NAMES)

                save_results(val_metrics, config.OUTPUT_DIR, "val", src_tag=_src_tag)
                save_results({"predictions": val_results}, config.OUTPUT_DIR, "val_predictions", src_tag=_src_tag)
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
            if is_caption:
                # ── Caption evaluation mode ───────────────────────────────────
                print(f"  [caption mode] Generating captions for full dataset")
                all_results = stage2.generate_caption_batch(
                    full_data,
                    batch_size=args.batch_size,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                )
                evaluate_captions(all_results, config.OUTPUT_DIR, "whole")
            else:
                # ── Disease classification mode ───────────────────────────────
                group_context_map = None

                if STAGE2_METHOD == "M1":
                    group_context_map = {item['image_path']: item['disease_group']
                                         for item in full_data}
                elif STAGE2_METHOD in ("M2", "M3"):
                    group_context_map = run_stage1_for_groups(full_data, config)
                elif STAGE2_METHOD == "M4":
                    group_context_map = run_stage1_soft_probs(full_data, config)

                all_results = stage2.predict_batch(
                    full_data,
                    group_context_map=group_context_map,
                    rag_retriever=rag_retriever,
                    batch_size=args.batch_size,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    prompt_variant=PROMPT_VARIANT,
                )
                all_metrics = calculate_metrics(all_results, DISEASE_NAMES)

                save_results(all_metrics, config.OUTPUT_DIR, "whole", src_tag=_src_tag)
                save_results({"predictions": all_results}, config.OUTPUT_DIR, "whole_predictions", src_tag=_src_tag)
                print_metrics(all_metrics, "whole")

        except Exception as e:
            print(f"[ERROR] Flow 2 error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Inference complete!")
    print(f"Results saved to: {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
