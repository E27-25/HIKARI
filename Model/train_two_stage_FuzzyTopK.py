"""
Two-Stage Training Pipeline for SkinCAP Dataset (FuzzyTopK Version)
Based on: OralGPT Paper (Zhang et al., arXiv:2510.13911v1)

Model: Qwen/Qwen3-VL-8B-Thinking
Framework: Unsloth

Enhancements over original train_two_stage.py:
- Fuzzy matching to consolidate near-duplicate disease labels
- Top-K class filtering (both stages train only on top K diseases)
- Stratified train/val split (maintains class proportions)
- Sqrt oversampling for class imbalance (train data only)

OUTPUT: 2 Models
- Model 1 (Stage 1): Classification
- Model 2 (Stage 2): Caption (trained from Stage 1 checkpoint)
"""

import os
import random
import math
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from datasets import Dataset
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

from thefuzz import process as fuzz_process
from sklearn.model_selection import train_test_split
import psutil

# Disable torch.compile
os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from unsloth import FastVisionModel, is_bf16_supported
from trl import SFTTrainer, SFTConfig


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Random seed
    SEED = 42

    # Model
    MODEL_NAME = "Qwen/Qwen3-VL-8B-Thinking"
    MAX_SEQ_LENGTH = 2048
    LOAD_IN_4BIT = True

    # LoRA (from paper)
    LORA_R = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0

    # Training
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    WARMUP_STEPS = 5
    WEIGHT_DECAY = 0.01

    # Epochs (from paper: 3 each)
    STAGE1_EPOCHS = 3
    STAGE2_EPOCHS = 3

    # Output directories (separate from original train_two_stage.py)
    OUTPUT_DIR_STAGE1 = "./skincap_fuzzytopk_classification"
    OUTPUT_DIR_STAGE1_MERGED = "./skincap_fuzzytopk_classification_merged"

    OUTPUT_DIR_STAGE2 = "./skincap_fuzzytopk_caption"
    OUTPUT_DIR_STAGE2_MERGED = "./skincap_fuzzytopk_caption_merged"

    # Data paths
    CSV_PATH = "./SkinCAP/skincap_v240623.csv"
    IMAGE_BASE_PATH = "./SkinCAP/skincap"

    # Fuzzy matching
    FUZZY_THRESHOLD = 91  # thefuzz threshold (0-100)

    # Top-K class filtering
    TOP_N_CLASSES = 10

    # Stratified split
    TEST_SIZE = 0.1

    # Class balancing: "sqrt", "inverse", or "none"
    BALANCE_METHOD = "sqrt"

    # Dataset processing
    DATASET_NUM_PROC = None  # None = auto-calculate, or set explicit value (1-64)

    # Split info persistence
    SPLIT_INFO_PATH = "./split_info_fuzzytopk.json"

    # Start Stage 1 from 3-stage group classifier instead of base model
    START_FROM_STAGE1_3STAGE: bool = False
    STAGE1_3STAGE_MERGED_PATH: str = "./skincap_3stage_group_classification_merged"

    # RAG-in-training: inject reference images in training prompts
    USE_RAG_IN_TRAINING: bool = False
    RAG_EXPERIMENT_TRAIN: str = "R2"   # R0-R4 encoder config
    RAG_K_TRAIN: int = 3               # references per query
    RAG_ALPHA_TRAIN: float = 0.9       # image weight in hybrid score

    # Stage 3 (caption) init weight:
    #   "checkpoint" (default) — load from Stage 1 LoRA checkpoint (train_from_lora)
    #   "merged"               — load from Stage 1 merged model (fresh LoRA on merged weights)
    STAGE3_INIT: str = "checkpoint"

    # STS: Selective Token Supervision for Stage 3 (caption)
    USE_STS: bool = False
    STS_IBR_BETA: float = 0.01         # IBR L2 penalty weight
    STS_DESCRIPTION_WEIGHT: float = 0.6  # w_ans for visual description sentence
    STS_DIAGNOSIS_WEIGHT: float = 1.0    # w_ans for diagnosis sentence
    STS_GAMMA: float = 0.1               # w_reason floor for non-medical tokens


# ============================================================================
# SEED
# ============================================================================

def set_seed(seed: int = Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed set to {seed}")


def calculate_dataset_num_proc(user_specified: int = None) -> int:
    """
    Calculate optimal num_proc for dataset.map() operations.

    Uses memory-aware calculation from UnslothSFTTrainer pattern.
    Higher CPU count = more parallelism, but limited by available memory.

    Args:
        user_specified: User override from Config.DATASET_NUM_PROC

    Returns:
        Optimal num_proc value (1-64)
    """
    import platform

    if user_specified is not None:
        num_proc = max(1, user_specified)
        if platform.system() == "Windows" and num_proc > 1:
            print(f"Windows detected: Attempting multiprocessing with num_proc={num_proc}")
            print("Note: If you encounter OSError, set --dataset_num_proc 1 to disable multiprocessing")
        return num_proc

    # Windows: Use conservative default but allow override
    if platform.system() == "Windows":
        # Try with 2 processes on Windows (more conservative than Linux)
        num_proc = min(2, psutil.cpu_count())
        print(f"Windows detected: Using num_proc={num_proc} for dataset formatting")
        print("If you encounter OSError, run with --dataset_num_proc 1")
        return num_proc

    # Linux/Mac: More aggressive parallelization
    # Start with CPU count + 4, capped at 64
    dataset_num_proc = min(max(psutil.cpu_count() + 4, 2), 64)

    # Memory-aware adjustment (from UnslothSFTTrainer.py:1024-1037)
    memory_gb_left = psutil.virtual_memory().available / (1024**3)

    if memory_gb_left <= 4:
        dataset_num_proc = 1  # Single-threaded for low memory
    elif memory_gb_left <= 6:
        dataset_num_proc = min(2, dataset_num_proc)
    elif memory_gb_left <= 8:
        dataset_num_proc = min(4, dataset_num_proc)
    elif memory_gb_left <= 12:
        dataset_num_proc = min(6, dataset_num_proc)

    return dataset_num_proc


def setup_environment():
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    set_seed(Config.SEED)
    print("Environment configured")


# ============================================================================
# PROMPTS
# ============================================================================

CLASSIFICATION_PROMPTS = [
    "What skin disease is shown in this image?",
    "Identify the dermatological condition in this photograph.",
    "What is the diagnosis for the skin lesion shown?",
]

CAPTION_PROMPTS = [
    "Describe this skin lesion image in detail. Include information about its appearance, possible diagnosis, and recommended examinations.",
    "Provide a clinical description of the skin condition shown in this image.",
    "What are the visual characteristics of this skin lesion? Describe its morphology and suggest possible diagnoses.",
]


# ============================================================================
# FUZZY MATCHING & TOP-K FILTERING
# ============================================================================

def norm_disease(text: str) -> str:
    """Normalize disease label: lowercase, strip, hyphens to spaces."""
    if not isinstance(text, str):
        return ""
    return text.lower().strip().replace("-", " ")


def fuzzy_consolidate_diseases(data: List[Dict], threshold: int = 91) -> List[Dict]:
    """
    Apply fuzzy matching to consolidate near-duplicate disease labels.

    Reuses the logic from EDA.ipynb:
    1. Normalize all disease names
    2. Use thefuzz to group similar names
    3. Map all variants to the first-seen canonical form
    """
    # Normalize
    for item in data:
        item["disease_original"] = item["disease"]
        item["disease"] = norm_disease(item["disease"])

    # Fuzzy grouping
    unique_diseases = list(set(item["disease"] for item in data))
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

    # Apply mapping
    for item in data:
        item["disease"] = mapping.get(item["disease"], item["disease"])

    n_before = len(unique_diseases)
    n_after = len(set(mapping.values()))
    n_merged = n_before - n_after
    print(f"Fuzzy consolidation (threshold={threshold}): "
          f"{n_before} unique labels -> {n_after} labels ({n_merged} merged)")

    return data


def display_and_filter_top_classes(
    data: List[Dict], top_n: int = 10
) -> Tuple[List[Dict], List[str]]:
    """
    Display ALL classes with counts, then filter to top N.

    Returns:
        filtered_data: Only samples from top N classes
        top_classes: Ordered list of top N class names
    """
    disease_counts = Counter(item["disease"] for item in data)

    # Display ALL classes
    print("\n" + "=" * 70)
    print(f"ALL DISEASE CLASSES ({len(disease_counts)} unique)")
    print("=" * 70)
    for rank, (disease, count) in enumerate(disease_counts.most_common(), 1):
        marker = "  <-- TOP-K" if rank <= top_n else ""
        print(f"  {rank:3d}. {disease:<50s} {count:4d}{marker}")

    # Get top N class names
    top_classes = [disease for disease, _ in disease_counts.most_common(top_n)]

    # Filter data
    filtered_data = [item for item in data if item["disease"] in top_classes]

    # Summary
    total_before = len(data)
    total_after = len(filtered_data)
    top_counts = Counter(item["disease"] for item in filtered_data)
    min_count = min(top_counts.values())
    max_count = max(top_counts.values())

    print(f"\n{'=' * 70}")
    print(f"SELECTED TOP {top_n} CLASSES FOR TRAINING")
    print(f"{'=' * 70}")
    print(f"  Samples: {total_before} -> {total_after} "
          f"({total_after / total_before * 100:.1f}% retained)")
    print(f"  Imbalance ratio: {max_count / min_count:.2f}:1 "
          f"(max={max_count}, min={min_count})")
    print()
    for i, disease in enumerate(top_classes, 1):
        count = top_counts[disease]
        bar = "#" * (count // 3)
        print(f"  {i:2d}. {disease:<50s} {count:4d}  {bar}")

    return filtered_data, top_classes


# ============================================================================
# STRATIFIED SPLIT
# ============================================================================

def stratified_split(
    data: List[Dict],
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stratified train/val split that maintains class proportions.
    """
    labels = [item["disease"] for item in data]
    indices = list(range(len(data)))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]

    # Verify stratification
    train_counts = Counter(item["disease"] for item in train_data)
    val_counts = Counter(item["disease"] for item in val_data)

    print(f"\nStratified Split (seed={seed}, test_size={test_size}):")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"\n  {'Class':<50s} {'Train':>6s} {'Val':>5s} {'Train%':>7s}")
    print(f"  {'-' * 70}")

    all_classes = sorted(set(labels), key=lambda x: -(train_counts.get(x, 0) + val_counts.get(x, 0)))
    for cls in all_classes:
        tr = train_counts.get(cls, 0)
        va = val_counts.get(cls, 0)
        ratio = tr / (tr + va) * 100 if (tr + va) > 0 else 0
        print(f"  {cls:<50s} {tr:6d} {va:5d} {ratio:6.1f}%")

    return train_data, val_data


# ============================================================================
# CLASS BALANCING (OVERSAMPLING)
# ============================================================================

def oversample_data(
    data: List[Dict],
    method: str = "sqrt",
    seed: int = 42,
) -> List[Dict]:
    """
    Oversample minority classes to reduce imbalance.

    Methods:
    - "sqrt":    target = max_count * sqrt(count / max_count)
                 Moderate balancing, reduces 2.2:1 -> ~1.5:1
    - "inverse": target = max_count for all classes (full equalization)
    - "none":    no oversampling, return data as-is

    Only apply to training data, never validation.
    """
    if method == "none":
        print("\nOversampling: disabled (method='none')")
        return data

    random.seed(seed)
    class_counts = Counter(item["disease"] for item in data)
    max_count = max(class_counts.values())

    # Compute target counts
    if method == "sqrt":
        targets = {
            cls: int(max_count * math.sqrt(count / max_count))
            for cls, count in class_counts.items()
        }
    elif method == "inverse":
        targets = {cls: max_count for cls in class_counts}
    else:
        raise ValueError(f"Unknown balance method: {method}")

    # Group data by class
    class_data = defaultdict(list)
    for item in data:
        class_data[item["disease"]].append(item)

    # Oversample
    balanced_data = []
    print(f"\nOversampling (method='{method}'):")
    print(f"  {'Class':<50s} {'Before':>7s} {'After':>7s} {'Action'}")
    print(f"  {'-' * 80}")

    for cls in sorted(class_counts, key=lambda x: -class_counts[x]):
        original = class_data[cls]
        target = targets[cls]
        current = len(original)

        if current >= target:
            balanced_data.extend(original)
            action = "kept"
        else:
            repeated = original.copy()
            while len(repeated) < target:
                repeated.append(random.choice(original))
            balanced_data.extend(repeated[:target])
            action = f"oversampled +{target - current}"

        print(f"  {cls:<50s} {current:7d} {target:7d}  {action}")

    random.shuffle(balanced_data)

    old_min = min(class_counts.values())
    old_max = max(class_counts.values())
    new_counts = Counter(item["disease"] for item in balanced_data)
    new_min = min(new_counts.values())
    new_max = max(new_counts.values())

    print(f"\n  Total: {len(data)} -> {len(balanced_data)}")
    print(f"  Imbalance ratio: {old_max / old_min:.2f}:1 -> {new_max / new_min:.2f}:1")

    return balanced_data


# ============================================================================
# SPLIT INFO PERSISTENCE
# ============================================================================

def save_split_info(
    train_data: List[Dict],
    val_data: List[Dict],
    top_classes: List[str],
):
    """Save split metadata for inference reproducibility."""
    info = {
        "top_classes": top_classes,
        "train_image_paths": [item["image_path"] for item in train_data],
        "val_image_paths": [item["image_path"] for item in val_data],
        "fuzzy_threshold": Config.FUZZY_THRESHOLD,
        "top_n": Config.TOP_N_CLASSES,
        "test_size": Config.TEST_SIZE,
        "seed": Config.SEED,
        "balance_method": Config.BALANCE_METHOD,
    }
    with open(Config.SPLIT_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"\nSplit info saved to: {Config.SPLIT_INFO_PATH}")


# ============================================================================
# DATA CONVERSION
# ============================================================================

def convert_classification(sample):
    """Stage 1: Classification format - loads image lazily from path.
    Supports optional RAG reference images via ref_paths/ref_labels columns."""
    image = Image.open(sample["image_path"]).convert("RGB")
    image.thumbnail((672, 672), Image.LANCZOS)

    content = []

    # Inject RAG reference images if present
    ref_paths = sample.get("ref_paths", []) or []
    ref_labels = sample.get("ref_labels", []) or []
    if ref_paths:
        content.append({"type": "text", "text": "Here are similar reference cases for context:"})
        for i, (rpath, rlabel) in enumerate(zip(ref_paths, ref_labels), 1):
            try:
                ref_img = Image.open(rpath).convert("RGB")
                ref_img.thumbnail((672, 672), Image.LANCZOS)
                content.append({"type": "image", "image": ref_img})
                content.append({"type": "text", "text": f"Reference {i}: {rlabel}"})
            except Exception:
                pass
        content.append({"type": "text", "text": "\nNow, identify the disease in this new image:"})

    content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": sample["prompt"]})

    conversation = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
    ]
    return {"messages": conversation}


def convert_caption(sample):
    """Stage 2: Caption format - loads image lazily from path"""
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
                {"type": "text", "text": sample["caption"]},
            ],
        },
    ]
    return {"messages": conversation}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_skincap_data() -> List[Dict]:
    """Load SkinCAP dataset - stores paths, not images (to save memory)"""
    print("Loading SkinCAP dataset...")

    csv_path = Path(Config.CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    df = df.dropna(subset=["skincap_file_path", "caption_zh_polish_en", "disease"])
    print(f"After filtering: {len(df)} valid rows")

    image_base = Path(Config.IMAGE_BASE_PATH)
    data_list = []

    for idx, row in df.iterrows():
        img_path = image_base / row["skincap_file_path"]

        if img_path.exists():
            data_list.append({
                "image_path": str(img_path),
                "disease": row["disease"],
                "caption": row["caption_zh_polish_en"],
            })

    print(f"Found {len(data_list)} valid samples")
    return data_list


def precompute_rag_refs(train_data: List[Dict]) -> Dict[str, list]:
    """Pre-compute RAG reference pairs for all training items.

    Must be called BEFORE loading the main model so GPU is free for the
    retrieval encoders. The retriever is destroyed afterwards to free VRAM.
    """
    import sys
    import torch
    import gc
    sys.path.insert(0, str(Path(__file__).parent))
    from rag_retrieval import HybridRAGRetriever, RAG_ENCODER_CONFIGS

    exp = Config.RAG_EXPERIMENT_TRAIN
    rag_cfg = RAG_ENCODER_CONFIGS[exp]
    index_path = f"./rag_index_{exp}_train.npz"
    print(f"\n[RAG-train] Loading/building index for {exp} ({index_path})...")

    retriever = HybridRAGRetriever.load_or_build(
        train_data, index_path,
        img_encoder_name=rag_cfg["img"],
        txt_encoder_name=rag_cfg["txt"],
        strategy=rag_cfg["strategy"],
    )
    retriever.alpha = Config.RAG_ALPHA_TRAIN

    refs_map: Dict[str, list] = {}
    print(f"[RAG-train] Pre-computing refs for {len(train_data)} training items...")
    for item in train_data:
        try:
            img = Image.open(item["image_path"]).convert("RGB")
            img.thumbnail((672, 672), Image.LANCZOS)
            raw_refs = retriever.retrieve(img, k=Config.RAG_K_TRAIN + 1)
            # Exclude self (same image path), take top K
            refs = [(p, l) for p, l, c in raw_refs if p != item["image_path"]][:Config.RAG_K_TRAIN]
            refs_map[item["image_path"]] = refs
        except Exception:
            refs_map[item["image_path"]] = []

    total_refs = sum(len(v) for v in refs_map.values())
    print(f"[RAG-train] Done. {total_refs} total ref pairs across {len(refs_map)} items.")

    del retriever
    torch.cuda.empty_cache()
    gc.collect()
    return refs_map


def prepare_classification_data(data: List[Dict], num_prompts: int = 3,
                                rag_refs_map: dict = None) -> Dataset:
    """Prepare data for Stage 1: Classification"""
    print("Preparing classification dataset...")

    samples = []
    for item in data:
        selected_prompts = random.sample(
            CLASSIFICATION_PROMPTS, min(num_prompts, len(CLASSIFICATION_PROMPTS))
        )
        # Get pre-computed RAG references (empty list if no RAG)
        refs = rag_refs_map.get(item["image_path"], []) if rag_refs_map else []
        ref_paths = [p for p, l in refs]
        ref_labels = [l for p, l in refs]

        for prompt in selected_prompts:
            samples.append({
                "image_path": item["image_path"],
                "prompt": prompt,
                "answer": f"This image shows {item['disease']}.",
                "ref_paths": ref_paths,
                "ref_labels": ref_labels,
            })

    random.shuffle(samples)
    rag_note = f" with RAG refs (K={Config.RAG_K_TRAIN})" if rag_refs_map else ""
    print(f"Created {len(samples)} classification samples{rag_note}")
    return Dataset.from_list(samples)


def prepare_caption_data(data: List[Dict], num_prompts: int = 3) -> Dataset:
    """Prepare data for Stage 2: Caption"""
    print("Preparing caption dataset...")

    samples = []
    for item in data:
        selected_prompts = random.sample(
            CAPTION_PROMPTS, min(num_prompts, len(CAPTION_PROMPTS))
        )
        for prompt in selected_prompts:
            samples.append({
                "image_path": item["image_path"],
                "prompt": prompt,
                "caption": item["caption"],
            })

    random.shuffle(samples)
    print(f"Created {len(samples)} caption samples")
    return Dataset.from_list(samples)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_base_model():
    """Load fresh base model for Stage 1"""
    print(f"\nLoading base model: {Config.MODEL_NAME}")

    model, tokenizer = FastVisionModel.from_pretrained(
        Config.MODEL_NAME,
        load_in_4bit=Config.LOAD_IN_4BIT,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        random_state=Config.SEED,
        use_rslora=False,
        loftq_config=None,
    )

    print(f"Model loaded with LoRA (r={Config.LORA_R}, alpha={Config.LORA_ALPHA})")
    return model, tokenizer


def load_from_checkpoint(checkpoint_path: str):
    """Load model from Stage 1 checkpoint for Stage 2

    Note: The checkpoint already has LoRA adapters, so we don't add them again.
    """
    print(f"\nLoading from checkpoint: {checkpoint_path}")

    model, tokenizer = FastVisionModel.from_pretrained(
        checkpoint_path,
        load_in_4bit=Config.LOAD_IN_4BIT,
        use_gradient_checkpointing="unsloth",
    )

    # Don't call get_peft_model() - the checkpoint already has LoRA adapters
    # Just enable training mode
    FastVisionModel.for_training(model)

    print(f"Model loaded from checkpoint (LoRA adapters already present)")
    return model, tokenizer


def load_from_merged_model(merged_model_path: str):
    """Load merged model from a previous stage and add fresh LoRA adapters."""
    print(f"\nLoading merged model: {merged_model_path}")

    model, tokenizer = FastVisionModel.from_pretrained(
        merged_model_path,
        load_in_4bit=Config.LOAD_IN_4BIT,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        random_state=Config.SEED,
        use_rslora=False,
        loftq_config=None,
    )

    print(f"Model loaded from merged weights + fresh LoRA (r={Config.LORA_R}, alpha={Config.LORA_ALPHA})")
    return model, tokenizer


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_stage1(train_data: List[Dict], val_data: List[Dict]):
    """
    Stage 1: Classification Training (with balanced oversampling)
    """
    print("\n" + "=" * 60)
    print("STAGE 1: CLASSIFICATION TRAINING")
    print("=" * 60)

    # Balance training data (NOT validation)
    balanced_train = oversample_data(
        train_data, method=Config.BALANCE_METHOD, seed=Config.SEED
    )

    # Pre-compute RAG refs BEFORE loading main model (frees GPU encoder first)
    rag_refs_map = None
    if Config.USE_RAG_IN_TRAINING:
        rag_refs_map = precompute_rag_refs(balanced_train)

    # Load model (base or from 3-stage group classifier)
    if Config.START_FROM_STAGE1_3STAGE:
        model, tokenizer = load_from_merged_model(Config.STAGE1_3STAGE_MERGED_PATH)
    else:
        model, tokenizer = load_base_model()

    # Prepare data (val uses no RAG refs — matches single-image inference eval)
    train_dataset = prepare_classification_data(balanced_train, num_prompts=3,
                                                rag_refs_map=rag_refs_map)
    val_dataset = prepare_classification_data(val_data, num_prompts=1)

    # Convert format (single-process when RAG refs are present to avoid memory issues)
    num_proc = calculate_dataset_num_proc(Config.DATASET_NUM_PROC)
    if rag_refs_map:
        num_proc = 1  # multi-image samples not safe with multiprocessing
    train_dataset = train_dataset.map(
        convert_classification,
        remove_columns=["image_path", "prompt", "answer", "ref_paths", "ref_labels"],
        desc="Formatting train data",
        num_proc=num_proc,
    )
    val_dataset = val_dataset.map(
        convert_classification,
        remove_columns=["image_path", "prompt", "answer", "ref_paths", "ref_labels"],
        desc="Formatting val data",
        num_proc=num_proc,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Training args
    training_args = SFTConfig(
        output_dir=Config.OUTPUT_DIR_STAGE1,
        num_train_epochs=Config.STAGE1_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=Config.WARMUP_STEPS,
        learning_rate=Config.LEARNING_RATE,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=Config.WEIGHT_DECAY,
        lr_scheduler_type="linear",
        seed=Config.SEED,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=Config.MAX_SEQ_LENGTH,
    )

    # Trainer
    from unsloth import UnslothVisionDataCollator

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    # Train
    print("\nTraining Stage 1...")
    trainer.train()

    # Save Model 1 (Classification)
    print("\n" + "=" * 60)
    print("SAVING MODEL 1: CLASSIFICATION")
    print("=" * 60)

    model.save_pretrained(Config.OUTPUT_DIR_STAGE1)
    tokenizer.save_pretrained(Config.OUTPUT_DIR_STAGE1)
    print(f"LoRA saved to: {Config.OUTPUT_DIR_STAGE1}")

    model.save_pretrained_merged(
        Config.OUTPUT_DIR_STAGE1_MERGED,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved to: {Config.OUTPUT_DIR_STAGE1_MERGED}")

    return model, tokenizer


def train_stage2(train_data: List[Dict], val_data: List[Dict]):
    """
    Stage 2: Caption Training (with balanced oversampling)

    Init weight determined by Config.STAGE3_INIT:
    - "checkpoint": load from Stage 1 LoRA checkpoint (default, continue adapters)
    - "merged":     load from Stage 1 merged model + fresh LoRA (Way 2)
    """
    print("\n" + "=" * 60)
    print("STAGE 2: CAPTION TRAINING")
    print(f"  Init mode: {Config.STAGE3_INIT}")
    if Config.USE_STS:
        print(f"  STS: ENABLED (IBR beta={Config.STS_IBR_BETA})")
    print("=" * 60)

    # Balance training data (NOT validation)
    balanced_train = oversample_data(
        train_data, method=Config.BALANCE_METHOD, seed=Config.SEED
    )

    # Load model according to init mode
    if Config.STAGE3_INIT == "merged":
        print(f"\n[Stage3 init=merged] Loading from merged model: {Config.OUTPUT_DIR_STAGE1_MERGED}")
        model, tokenizer = load_from_merged_model(Config.OUTPUT_DIR_STAGE1_MERGED)
    else:
        # Default: load from LoRA checkpoint
        model, tokenizer = load_from_checkpoint(Config.OUTPUT_DIR_STAGE1)

    # Prepare data
    train_dataset = prepare_caption_data(balanced_train, num_prompts=3)
    val_dataset = prepare_caption_data(val_data, num_prompts=1)

    # Convert format
    num_proc = calculate_dataset_num_proc(Config.DATASET_NUM_PROC)
    train_dataset = train_dataset.map(
        convert_caption,
        remove_columns=["image_path", "prompt", "caption"],
        desc="Formatting train data",
        num_proc=num_proc,
    )
    val_dataset = val_dataset.map(
        convert_caption,
        remove_columns=["image_path", "prompt", "caption"],
        desc="Formatting val data",
        num_proc=num_proc,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Training args
    training_args = SFTConfig(
        output_dir=Config.OUTPUT_DIR_STAGE2,
        num_train_epochs=Config.STAGE2_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=Config.WARMUP_STEPS,
        learning_rate=Config.LEARNING_RATE,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=Config.WEIGHT_DECAY,
        lr_scheduler_type="linear",
        seed=Config.SEED,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=Config.MAX_SEQ_LENGTH,
    )

    # Trainer
    from unsloth import UnslothVisionDataCollator

    if Config.USE_STS:
        # Build STS-augmented trainer with medical token importance
        from medical_token_importance import (
            MedicalSTSConfig, MedicalTokenImportanceScorer, STSSFTTrainer
        )

        sts_config = MedicalSTSConfig(
            description_weight=Config.STS_DESCRIPTION_WEIGHT,
            diagnosis_weight=Config.STS_DIAGNOSIS_WEIGHT,
            recommendation_weight=Config.STS_DIAGNOSIS_WEIGHT,
            ibr_beta=Config.STS_IBR_BETA,
            gamma=Config.STS_GAMMA,
        )
        sts_scorer = MedicalTokenImportanceScorer(sts_config, tokenizer)

        class _STSSFTTrainer(STSSFTTrainer, SFTTrainer):
            pass

        trainer = _STSSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
        )
        trainer.setup_sts(sts_scorer, sts_config)
    else:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
        )

    # Train
    print("\nTraining Stage 2...")
    trainer.train()

    # Save Model 2 (Caption)
    print("\n" + "=" * 60)
    print("SAVING MODEL 2: CAPTION")
    print("=" * 60)

    model.save_pretrained(Config.OUTPUT_DIR_STAGE2)
    tokenizer.save_pretrained(Config.OUTPUT_DIR_STAGE2)
    print(f"LoRA saved to: {Config.OUTPUT_DIR_STAGE2}")

    model.save_pretrained_merged(
        Config.OUTPUT_DIR_STAGE2_MERGED,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved to: {Config.OUTPUT_DIR_STAGE2_MERGED}")

    return model, tokenizer


# ============================================================================
# DATA PREPARATION (shared across modes)
# ============================================================================

def prepare_data_pipeline() -> Tuple[List[Dict], List[Dict], List[str]]:
    """
    Shared data pipeline: load -> fuzzy -> filter top-K -> stratified split.
    Returns (train_data, val_data, top_classes).
    """
    # [1/4] Load data
    print("\n[1/4] Loading data...")
    data = load_skincap_data()

    # [2/4] Fuzzy consolidation
    print("\n[2/4] Consolidating disease labels (fuzzy matching)...")
    data = fuzzy_consolidate_diseases(data, threshold=Config.FUZZY_THRESHOLD)

    # [3/4] Display all classes and filter to top K
    print("\n[3/4] Filtering to top-K classes...")
    data, top_classes = display_and_filter_top_classes(data, top_n=Config.TOP_N_CLASSES)

    # [4/4] Stratified split
    print("\n[4/4] Stratified train/val split...")
    train_data, val_data = stratified_split(
        data, test_size=Config.TEST_SIZE, seed=Config.SEED
    )

    # Save split info for inference reproducibility
    save_split_info(train_data, val_data, top_classes)

    return train_data, val_data, top_classes


# ============================================================================
# MAIN
# ============================================================================

def train_two_stage():
    """Full two-stage training pipeline with FuzzyTopK"""

    print("=" * 60)
    print("TWO-STAGE TRAINING PIPELINE (FuzzyTopK)")
    print("=" * 60)
    print(f"Model:            {Config.MODEL_NAME}")
    print(f"Seed:             {Config.SEED}")
    print(f"Stage 1 epochs:   {Config.STAGE1_EPOCHS}")
    print(f"Stage 2 epochs:   {Config.STAGE2_EPOCHS}")
    print(f"Fuzzy threshold:  {Config.FUZZY_THRESHOLD}")
    print(f"Top-K classes:    {Config.TOP_N_CLASSES}")
    print(f"Balance method:   {Config.BALANCE_METHOD}")
    print("=" * 60)

    setup_environment()

    # Data pipeline
    train_data, val_data, top_classes = prepare_data_pipeline()
    print(f"\nTrain: {len(train_data)}, Val: {len(val_data)}")

    # Stage 1: Classification
    print("\n[5/6] Stage 1: Classification...")
    train_stage1(train_data, val_data)

    # Stage 2: Caption (loads from Stage 1)
    print("\n[6/6] Stage 2: Caption...")
    train_stage2(train_data, val_data)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"""
OUTPUT MODELS (FuzzyTopK - Top {Config.TOP_N_CLASSES} classes, {Config.BALANCE_METHOD} balanced):

  MODEL 1: CLASSIFICATION
    LoRA:   {Config.OUTPUT_DIR_STAGE1}
    Merged: {Config.OUTPUT_DIR_STAGE1_MERGED}

  MODEL 2: CAPTION
    LoRA:   {Config.OUTPUT_DIR_STAGE2}
    Merged: {Config.OUTPUT_DIR_STAGE2_MERGED}

  Split info: {Config.SPLIT_INFO_PATH}

  Top-K classes trained on:
{chr(10).join(f'    {i+1:2d}. {c}' for i, c in enumerate(top_classes))}
""")


def train_stage1_only():
    """Train only Stage 1 with FuzzyTopK pipeline"""
    setup_environment()
    train_data, val_data, top_classes = prepare_data_pipeline()
    train_stage1(train_data, val_data)


def train_stage2_only():
    """Train only Stage 2 (requires Stage 1 checkpoint)"""
    if not Path(Config.OUTPUT_DIR_STAGE1).exists():
        print(f"Error: Stage 1 checkpoint not found at {Config.OUTPUT_DIR_STAGE1}")
        print("Please run Stage 1 first.")
        return

    setup_environment()
    train_data, val_data, top_classes = prepare_data_pipeline()
    train_stage2(train_data, val_data)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Two-Stage Training with Fuzzy Matching & Top-K Class Balancing"
    )
    parser.add_argument("--mode", type=str, default="both",
                        choices=["both", "stage1", "stage2"],
                        help="Training mode")
    parser.add_argument("--stage1_epochs", type=int, default=3)
    parser.add_argument("--stage2_epochs", type=int, default=3)
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of top classes to keep")
    parser.add_argument("--balance", type=str, default="sqrt",
                        choices=["sqrt", "inverse", "none"],
                        help="Class balancing method")
    parser.add_argument("--fuzzy_threshold", type=int, default=91,
                        help="Fuzzy matching threshold (0-100)")
    parser.add_argument("--dataset_num_proc", type=int, default=None,
                        help="Number of processes for dataset formatting (default: auto)")
    parser.add_argument("--start_from_stage1", action="store_true", default=False,
                        help="Start Stage 1 from 3-stage group classifier merged model instead of base model")
    parser.add_argument("--use_rag_in_training", action="store_true", default=False,
                        help="Inject RAG reference images in training prompts (Experiment E)")
    parser.add_argument("--rag_exp_train", type=str, default="R2",
                        choices=["R0", "R1", "R2", "R3", "R4"],
                        help="RAG encoder config for training references (default: R2)")
    parser.add_argument("--rag_alpha_train", type=float, default=0.9,
                        help="Alpha (image weight) for RAG retrieval during training (default: 0.9)")
    parser.add_argument("--rag_k_train", type=int, default=None,
                        help="Override RAG_K_TRAIN (number of reference images per training sample, default: Config value 3)")
    parser.add_argument("--stage3_init", type=str, default="checkpoint",
                        choices=["checkpoint", "merged"],
                        help="Stage 3 (caption) init: 'checkpoint' = load from Stage 1 LoRA (default/Way 1); "
                             "'merged' = load from Stage 1 merged model with fresh LoRA (Way 2)")
    parser.add_argument("--use_sts", action="store_true", default=False,
                        help="Enable Selective Token Supervision (STS) + IBR for Stage 3 caption training")
    parser.add_argument("--sts_beta", type=float, default=0.01,
                        help="IBR regularization beta for STS (default: 0.01)")
    parser.add_argument("--sts_desc_weight", type=float, default=0.6,
                        help="STS sentence weight for visual description (default: 0.6)")
    parser.add_argument("--sts_diag_weight", type=float, default=1.0,
                        help="STS sentence weight for diagnosis/recommendation (default: 1.0)")

    args = parser.parse_args()

    Config.STAGE1_EPOCHS = args.stage1_epochs
    Config.STAGE2_EPOCHS = args.stage2_epochs
    Config.TOP_N_CLASSES = args.top_n
    Config.BALANCE_METHOD = args.balance
    Config.FUZZY_THRESHOLD = args.fuzzy_threshold
    Config.DATASET_NUM_PROC = args.dataset_num_proc
    Config.START_FROM_STAGE1_3STAGE = args.start_from_stage1
    Config.USE_RAG_IN_TRAINING = args.use_rag_in_training
    Config.RAG_EXPERIMENT_TRAIN = args.rag_exp_train
    Config.RAG_ALPHA_TRAIN = args.rag_alpha_train
    if args.rag_k_train is not None:
        Config.RAG_K_TRAIN = args.rag_k_train

    # STS flags
    Config.STAGE3_INIT = args.stage3_init
    Config.USE_STS = args.use_sts
    Config.STS_IBR_BETA = args.sts_beta
    Config.STS_DESCRIPTION_WEIGHT = args.sts_desc_weight
    Config.STS_DIAGNOSIS_WEIGHT = args.sts_diag_weight

    if Config.START_FROM_STAGE1_3STAGE:
        Config.OUTPUT_DIR_STAGE1 = "./skincap_fuzzytopk_s1cascade_classification"
        Config.OUTPUT_DIR_STAGE1_MERGED = "./skincap_fuzzytopk_s1cascade_classification_merged"
        # Update Stage 2 output dirs to reflect the s1cascade init
        init_tag = "_merged_init" if Config.STAGE3_INIT == "merged" else ""
        sts_tag = "_sts" if Config.USE_STS else ""
        Config.OUTPUT_DIR_STAGE2 = (
            f"./skincap_stage3_caption_fuzzytopk_s1cascade{init_tag}{sts_tag}_classification"
        )
        Config.OUTPUT_DIR_STAGE2_MERGED = (
            f"./skincap_stage3_caption_fuzzytopk_s1cascade{init_tag}{sts_tag}_classification_merged"
        )

    if Config.USE_RAG_IN_TRAINING:
        alpha_tag = str(Config.RAG_ALPHA_TRAIN).replace(".", "")
        rag_suffix = f"_rag{Config.RAG_EXPERIMENT_TRAIN}_a{alpha_tag}"
        base = "skincap_fuzzytopk_s1cascade" if Config.START_FROM_STAGE1_3STAGE else "skincap_fuzzytopk"
        Config.OUTPUT_DIR_STAGE1 = f"./{base}{rag_suffix}_classification"
        Config.OUTPUT_DIR_STAGE1_MERGED = f"./{base}{rag_suffix}_classification_merged"

    if args.mode == "both":
        train_two_stage()
    elif args.mode == "stage1":
        train_stage1_only()
    elif args.mode == "stage2":
        train_stage2_only()
