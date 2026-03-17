"""
Three-Stage Training Pipeline for SkinCAP Dataset (Strategy G - Top-15 Simplified)
Based on: OralGPT Paper (Zhang et al., arXiv:2510.13911v1)

Model: Qwen/Qwen3-VL-8B-Thinking
Framework: Unsloth

Strategy G - Top-15 Disease Simplified Pipeline:
- Uses only the top 15 most frequent diseases (~1,355 samples)
- Auto-derives groups from disease-to-group mapping (4 or 3 groups depending on GROUP_MODE)
- All Stage 2 disease labels are named (no "Other_GroupX" fallback)
- Two GROUP_MODE options to compare group separation approaches:
    "4group": Inflammatory, Benign, Malignant, Acne (4 classes, 5:1 imbalance)
    "3group": Inflammatory, Benign+Acne merged, Malignant (3 classes, 2:1 imbalance)

OUTPUT: 3 Models
- Model 1 (Stage 1): Group Classification (3 or 4 classes)
- Model 2 (Stage 2): Disease Classification (all 15 named diseases per group)
- Model 3 (Stage 3): Caption Generation (trained from Stage 2 checkpoint)
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

# Heavy model dependencies are imported inside functions to avoid loading them
# when only data processing functions are needed (e.g., for testing)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Random seed
    SEED = 42

    # -----------------------------------------------------------------------
    # Strategy G: Group mode — switch between "4group" and "3group" to compare
    # "4group": Inflammatory / Benign / Malignant / Acne  (4 classes)
    # "3group": Inflammatory / Benign+Acne merged / Malignant  (3 classes)
    # -----------------------------------------------------------------------
    # GROUP_MODE = "3group"  # "4group" or "3group"
    # TOP_N = 10             # 15 = top-15 diseases, 10 = top-10 diseases

    # GROUP_MODE = "3group"
    # TOP_N = 10

    GROUP_MODE = "4group"
    TOP_N = 10

    # Model
    MODEL_NAME = "Qwen/Qwen3-VL-8B-Thinking"
    MAX_SEQ_LENGTH = 2048  # Full sequence length (memory managed)
    LOAD_IN_4BIT = True

    # LoRA (from paper)
    LORA_R = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.0

    # Training
    BATCH_SIZE = 2  # Full batch size (memory managed)
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 8
    LEARNING_RATE = 2e-4
    WARMUP_STEPS = 5
    WEIGHT_DECAY = 0.01

    # Epochs
    STAGE1_EPOCHS = 5  # Increased: group classification needs more training
    STAGE2_EPOCHS = 3
    STAGE3_EPOCHS = 3

    # Output directories for 3-stage pipeline
    OUTPUT_DIR_STAGE1 = "./skincap_3stage_group_classification"
    OUTPUT_DIR_STAGE1_MERGED = "./skincap_3stage_group_classification_merged"

    OUTPUT_DIR_STAGE2 = "./skincap_3stage_disease_classification"
    OUTPUT_DIR_STAGE2_MERGED = "./skincap_3stage_disease_classification_merged"

    OUTPUT_DIR_STAGE3 = "./skincap_3stage_caption"
    OUTPUT_DIR_STAGE3_MERGED = "./skincap_3stage_caption_merged"

    # Stage 3 source model path — which model to load from for Stage 3 training.
    # Empty string = use OUTPUT_DIR_STAGE2_MERGED (default 3-stage pipeline).
    # Override via --stage3_source to use a different base (e.g., fuzzytopk_s1cascade).
    STAGE3_SOURCE_MODEL: str = ""
    # Stage 3 learning rate override. 0.0 = use global LEARNING_RATE.
    # Use lower value (e.g. 5e-5) to prevent catastrophic forgetting when
    # starting from a well-trained model like fuzzytopk_s1cascade.
    STAGE3_LR: float = 0.0

    # Data paths
    CSV_PATH = "./SkinCAP/skincap_v240623.csv"
    IMAGE_BASE_PATH = "./SkinCAP/skincap"

    # Fuzzy matching
    FUZZY_THRESHOLD = 91  # thefuzz threshold (0-100)

    # Strategy G - Top-15 diseases, K = all diseases per group (no "Other" needed)
    # 4-group mode
    TOP_K_CONFIG_4 = {
        '1. Inflammatory & Autoimmune Diseases': 8,  # 8 diseases: psoriasis, lupus, ...
        '2. Benign Tumors, Nevi & Cysts': 2,         # 2 diseases: melanocytic nevi, seborrheic keratosis
        '3. Malignant Skin Tumors': 3,                # 3 diseases: SCC, BCC, mycosis fungoides
        '4. Acne & Follicular Disorders': 2,          # 2 diseases: acne vulgaris, folliculitis
    }
    # 3-group mode (acne merged into benign)
    TOP_K_CONFIG_3 = {
        '1. Inflammatory & Autoimmune Diseases': 8,  # 8 diseases
        '2. Benign & Other Non-Malignant': 4,         # 4 diseases: melanocytic nevi, seborrheic keratosis, acne vulgaris, folliculitis
        '3. Malignant Skin Tumors': 3,                # 3 diseases
    }
    # Top-10 K configs
    TOP_K_CONFIG_4_TOP10 = {
        '1. Inflammatory & Autoimmune Diseases': 6,  # 6 diseases
        '2. Benign Tumors, Nevi & Cysts': 1,          # 1 disease: melanocytic nevi
        '3. Malignant Skin Tumors': 2,                 # 2 diseases: SCC, BCC
        '4. Acne & Follicular Disorders': 1,           # 1 disease: acne vulgaris
    }
    TOP_K_CONFIG_3_TOP10 = {
        '1. Inflammatory & Autoimmune Diseases': 6,  # 6 diseases
        '2. Benign & Other Non-Malignant': 2,          # 2 diseases: melanocytic nevi, acne vulgaris
        '3. Malignant Skin Tumors': 2,                 # 2 diseases: SCC, BCC
    }
    # Active config — set at module level based on GROUP_MODE + TOP_N
    TOP_K_CONFIG = TOP_K_CONFIG_4  # default; overridden below after class definition

    # Stage 2 method: controls group context in disease classification prompts
    # "M0": no group context (baseline)
    # "M1": GT group at both train and inference (oracle upper bound)
    # "M2": GT group at train, Stage 1 predicted group at inference
    # "M3": Stage 1 predicted group at train, Stage 1 predicted group at inference
    # "M4": GT group at train, Stage 1 top-2 beam candidates + scores at inference
    STAGE2_METHOD = "M0"

    # When True, Stage 2 loads the BASE model instead of Stage 1 merged weights
    # Eliminates the cascade penalty (~22%) — use for M1_base, M3_base etc.
    USE_BASE_MODEL_FOR_STAGE2: bool = False

    # When True, Stage 2 uses fuzzytopk train images (SCCIS class) instead of 3stage train images
    # Val split stays as 3stage val (101 samples) for fair comparison
    USE_FUZZYTOPK_SPLIT_FOR_STAGE2: bool = False

    # Path to cache Stage 1 predictions on training set (used by M3)
    STAGE1_TRAIN_PREDICTIONS_PATH = "./stage1_train_predictions.json"

    # Stratified split
    TEST_SIZE = 0.1

    # Class balancing: "sqrt", "inverse", or "none"
    BALANCE_METHOD = "sqrt"

    # Dataset processing
    DATASET_NUM_PROC = None  # None = auto-calculate, or set explicit value (1-64)

    # Split info persistence
    SPLIT_INFO_PATH = "./split_info_3stage.json"

    # Image augmentation settings (probabilistic — 50% chance per transform)
    USE_AUGMENTATION = True  # Enabled with memory management
    AUGMENT_ROTATION = 15    # Random rotation degrees (±15°), applied with 50% probability
    AUGMENT_BRIGHTNESS = 0.25 # Brightness jitter factor, applied with 50% probability
    AUGMENT_CONTRAST = 0.25   # Contrast jitter factor, applied with 50% probability
    AUGMENT_SATURATION = 0.25 # Saturation jitter factor, applied with 50% probability
    HORIZONTAL_FLIP_PROB = 0.5  # Probability of horizontal flip
    # No random crop — can cut out the lesion in medical images

    # Memory management settings (for preventing OOM during training)
    ENABLE_MEMORY_MANAGEMENT = True  # Enable aggressive memory cleanup
    TORCH_EMPTY_CACHE_STEPS = 50     # Clear CUDA cache every N steps (default: 250)
    MANUAL_CLEANUP_INTERVAL = 100    # Manual cleanup every N steps (gc.collect)


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


def calculate_dataset_num_proc(user_specified: int = None):
    """
    Calculate optimal num_proc for dataset.map() operations.

    On Windows, returns None to disable multiprocessing due to pipe size limitations
    with PIL Image objects. On Linux/Mac, uses memory-aware parallelization.

    Args:
        user_specified: User override from Config.DATASET_NUM_PROC

    Returns:
        int for Linux/Mac (1-64), None for Windows (disables multiprocessing)
    """
    import platform

    # Windows: Disable multiprocessing due to pipe size limits with PIL Images
    if platform.system() == "Windows":
        if user_specified is not None and user_specified > 1:
            print("=" * 70)
            print("[WARNING] Windows multiprocessing disabled")
            print("=" * 70)
            print("Windows cannot handle multiprocessing with PIL Image objects")
            print("due to named pipe buffer size limitations.")
            print(f"Your request for num_proc={user_specified} is ignored.")
            print("Dataset formatting will run single-threaded (num_proc=None).")
            print("=" * 70)
        return None  # Disable multiprocessing on Windows

    # User override (Linux/Mac only)
    if user_specified is not None:
        return max(1, user_specified)

    # Linux/Mac: Memory-aware parallelization
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

# Stage 1: Group Classification — prompts must list EXACT numbered group names
# (training answers are the group names verbatim; prompts must match)

# 4-group mode prompts
_GROUP_PROMPTS_4 = [
    "Classify this skin condition into one of these 4 medical categories:\n"
    "1. Inflammatory & Autoimmune Diseases\n"
    "2. Benign Tumors, Nevi & Cysts\n"
    "3. Malignant Skin Tumors\n"
    "4. Acne & Follicular Disorders\n\n"
    "Answer with only the category number and name.",

    "What medical group does this skin lesion belong to?\n"
    "1. Inflammatory & Autoimmune Diseases\n"
    "2. Benign Tumors, Nevi & Cysts\n"
    "3. Malignant Skin Tumors\n"
    "4. Acne & Follicular Disorders\n\n"
    "Answer with only the category number and name.",

    "Identify which medical category this skin condition belongs to:\n"
    "1. Inflammatory & Autoimmune Diseases\n"
    "2. Benign Tumors, Nevi & Cysts\n"
    "3. Malignant Skin Tumors\n"
    "4. Acne & Follicular Disorders\n\n"
    "Answer with only the category number and name.",
]

# 3-group mode prompts
_GROUP_PROMPTS_3 = [
    "Classify this skin condition into one of these 3 medical categories:\n"
    "1. Inflammatory & Autoimmune Diseases\n"
    "2. Benign & Other Non-Malignant\n"
    "3. Malignant Skin Tumors\n\n"
    "Answer with only the category number and name.",

    "What medical group does this skin lesion belong to?\n"
    "1. Inflammatory & Autoimmune Diseases\n"
    "2. Benign & Other Non-Malignant\n"
    "3. Malignant Skin Tumors\n\n"
    "Answer with only the category number and name.",

    "Identify which medical category this skin condition belongs to:\n"
    "1. Inflammatory & Autoimmune Diseases\n"
    "2. Benign & Other Non-Malignant\n"
    "3. Malignant Skin Tumors\n\n"
    "Answer with only the category number and name.",
]

GROUP_CLASSIFICATION_PROMPTS = _GROUP_PROMPTS_4 if Config.GROUP_MODE == "4group" else _GROUP_PROMPTS_3

# Stage 2: Disease Classification
DISEASE_CLASSIFICATION_PROMPTS = [
    "Carefully examine this dermatological image. Look for: lesion morphology (papule/plaque/macule/nodule), color (red/violet/white/brown/black), scale or crust, border sharpness, and distribution. Based on these visual features, what is the specific skin disease?",
    "Identify the dermatological condition in this photograph.",
    "What is the diagnosis for the skin lesion shown?",
]

# Stage 2: Disease classification WITH group context (M1/M2/M3/M4)
# {group} placeholder is filled with the actual group name at data-prep time
DISEASE_PROMPTS_WITH_GROUP = [
    "This skin lesion belongs to the group '{group}'. Examine the lesion morphology (papules, plaques, macules), color (red, violet, white, brown), scale/crust, border sharpness, and distribution pattern. Based on these visual features, what is the specific skin disease?",
    "Given this condition is classified as '{group}', identify the specific dermatological disease.",
    "This image shows a condition in the '{group}' category. What is the precise diagnosis?",
]

# Stage 3: Caption Generation
CAPTION_PROMPTS = [
    "Describe this skin lesion image in detail. Include information about its appearance, possible diagnosis, and recommended examinations.",
    "Provide a clinical description of the skin condition shown in this image.",
    "What are the visual characteristics of this skin lesion? Describe its morphology and suggest possible diagnoses.",
]


# ============================================================================
# NORMALIZATION
# ============================================================================

def norm_disease(text: str) -> str:
    """Normalize disease label: lowercase, strip, hyphens to spaces."""
    if not isinstance(text, str):
        return ""
    return text.lower().strip().replace("-", " ")


# ============================================================================
# GROUP CATEGORIZATION (Strategy G - Top-15 Simplified)
# ============================================================================

# Top 15 most frequent diseases (~1,355 samples total)
TOP_15_DISEASES = {
    "squamous cell carcinoma", "basal cell carcinoma", "psoriasis",
    "melanocytic nevi", "lupus erythematosus", "lichen planus",
    "scleroderma", "photodermatoses", "acne vulgaris", "sarcoidosis",
    "seborrheic keratosis", "allergic contact dermatitis",
    "neutrophilic dermatoses", "mycosis fungoides", "folliculitis",
}

# Top 10 most frequent diseases (~1,016 samples total)
# Removes: seborrheic keratosis, allergic contact dermatitis,
#          neutrophilic dermatoses, mycosis fungoides, folliculitis
TOP_10_DISEASES = {
    "squamous cell carcinoma", "basal cell carcinoma", "psoriasis",
    "melanocytic nevi", "lupus erythematosus", "lichen planus",
    "scleroderma", "photodermatoses", "acne vulgaris", "sarcoidosis",
}

# 4-group disease mapping (Inflammatory / Benign / Malignant / Acne)
DISEASE_GROUPS_4 = {
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

# 3-group disease mapping (Acne merged into Benign for better balance)
DISEASE_GROUPS_3 = {
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

# Top-10 group dicts (same group names, fewer diseases)
DISEASE_GROUPS_4_TOP10 = {
    "1. Inflammatory & Autoimmune Diseases": [
        "psoriasis", "lupus erythematosus", "lichen planus", "scleroderma",
        "photodermatoses", "sarcoidosis",
    ],
    "2. Benign Tumors, Nevi & Cysts": ["melanocytic nevi"],
    "3. Malignant Skin Tumors": ["squamous cell carcinoma in situ", "basal cell carcinoma"],
    "4. Acne & Follicular Disorders": ["acne vulgaris"],
}
DISEASE_GROUPS_3_TOP10 = {
    "1. Inflammatory & Autoimmune Diseases": [
        "psoriasis", "lupus erythematosus", "lichen planus", "scleroderma",
        "photodermatoses", "sarcoidosis",
    ],
    "2. Benign & Other Non-Malignant": ["melanocytic nevi", "acne vulgaris"],
    "3. Malignant Skin Tumors": ["squamous cell carcinoma in situ", "basal cell carcinoma"],
}

# ============================================================================
# MODE-BASED CONFIG OVERRIDE (must be after all disease constants are defined)
# ============================================================================
ACTIVE_TOP_DISEASES = TOP_15_DISEASES if Config.TOP_N == 15 else TOP_10_DISEASES

if Config.GROUP_MODE == "4group":
    if Config.TOP_N == 15:
        DISEASE_GROUPS = DISEASE_GROUPS_4
        Config.TOP_K_CONFIG = Config.TOP_K_CONFIG_4
    else:
        DISEASE_GROUPS = DISEASE_GROUPS_4_TOP10
        Config.TOP_K_CONFIG = Config.TOP_K_CONFIG_4_TOP10
else:  # 3group
    if Config.TOP_N == 15:
        DISEASE_GROUPS = DISEASE_GROUPS_3
        Config.TOP_K_CONFIG = Config.TOP_K_CONFIG_3
    else:
        DISEASE_GROUPS = DISEASE_GROUPS_3_TOP10
        Config.TOP_K_CONFIG = Config.TOP_K_CONFIG_3_TOP10

Config.OUTPUT_DIR_STAGE1 = f"./skincap_3stage_group_{Config.GROUP_MODE}_top{Config.TOP_N}"
Config.OUTPUT_DIR_STAGE1_MERGED = f"./skincap_3stage_group_{Config.GROUP_MODE}_top{Config.TOP_N}_merged"
Config.OUTPUT_DIR_STAGE2 = f"./skincap_3stage_disease_{Config.STAGE2_METHOD}"
Config.OUTPUT_DIR_STAGE2_MERGED = f"./skincap_3stage_disease_{Config.STAGE2_METHOD}_merged"

# Build reverse lookup: disease_name -> group_name
DISEASE_TO_GROUP = {}
for _group, _diseases in DISEASE_GROUPS.items():
    for _disease in _diseases:
        DISEASE_TO_GROUP[norm_disease(_disease)] = _group


def categorize_morphology(disease):
    """
    Map disease label to one of the active medical groups (Strategy G).

    Only covers top-N diseases. Samples not in ACTIVE_TOP_DISEASES are filtered out
    before this function is called.

    Args:
        disease: Disease name (string)

    Returns:
        Group name (e.g., "1. Inflammatory & Autoimmune Diseases") or "Unknown"
    """
    return DISEASE_TO_GROUP.get(norm_disease(disease), "Unknown")


# ============================================================================
# FUZZY MATCHING (from train_two_stage_FuzzyTopK.py)
# ============================================================================

def fuzzy_consolidate_diseases(data: List[Dict], threshold: int = 91) -> List[Dict]:
    """
    Apply fuzzy matching to consolidate near-duplicate disease labels.

    Args:
        data: List of dicts with 'disease' field
        threshold: Fuzzy matching threshold (0-100)

    Returns:
        Modified data with consolidated disease names
    """
    # Normalize
    for item in data:
        item["disease_original"] = item["disease"]
        item["disease"] = norm_disease(item["disease"])

    # Fuzzy grouping
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

    # Apply mapping
    for item in data:
        item["disease"] = mapping.get(item["disease"], item["disease"])

    n_before = len(unique_diseases)
    n_after = len(set(mapping.values()))
    n_merged = n_before - n_after
    print(f"Fuzzy consolidation (threshold={threshold}): "
          f"{n_before} unique labels -> {n_after} labels ({n_merged} merged)")

    return data


# ============================================================================
# IMAGE AUGMENTATION
# ============================================================================

def apply_image_augmentation(image: Image.Image, apply_aug: bool = True) -> Image.Image:
    """
    Apply medical image augmentation to PIL Image.

    Probabilistic augmentation for dermatology images (50% chance per transform):
    - Random rotation (±15°) — skin lesions can appear at any angle
    - Color jitter (brightness/contrast/saturation) — lighting variations
    - Horizontal flip — lesions are roughly symmetric
    No random crop — cropping can remove the lesion from frame.

    Args:
        image: PIL Image in RGB format
        apply_aug: Whether to apply augmentation (False for validation)

    Returns:
        Augmented PIL Image
    """
    if not apply_aug or not Config.USE_AUGMENTATION:
        return image

    import random
    from PIL import ImageEnhance

    # Random rotation (50% chance)
    if random.random() < 0.5:
        angle = random.uniform(-Config.AUGMENT_ROTATION, Config.AUGMENT_ROTATION)
        image = image.rotate(angle, fillcolor=(128, 128, 128))

    # Horizontal flip (50% chance)
    if random.random() < Config.HORIZONTAL_FLIP_PROB:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Brightness adjustment (50% chance)
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(1 - Config.AUGMENT_BRIGHTNESS, 1 + Config.AUGMENT_BRIGHTNESS)
        image = enhancer.enhance(factor)

    # Contrast adjustment (50% chance)
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(1 - Config.AUGMENT_CONTRAST, 1 + Config.AUGMENT_CONTRAST)
        image = enhancer.enhance(factor)

    # Color/saturation adjustment (50% chance)
    if random.random() < 0.5:
        enhancer = ImageEnhance.Color(image)
        factor = random.uniform(1 - Config.AUGMENT_SATURATION, 1 + Config.AUGMENT_SATURATION)
        image = enhancer.enhance(factor)

    return image


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def cleanup_memory():
    """
    Aggressive memory cleanup to prevent OOM during long training runs.

    Based on patterns from UnslothPPOTrainer.py (lines 1021-1023, 1186-1188).
    Combines CUDA cache clearing with Python garbage collection.
    """
    import gc
    import torch

    if not Config.ENABLE_MEMORY_MANAGEMENT:
        return

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Force Python garbage collection
    gc.collect()


class MemoryCleanupCallback:
    """
    Callback to perform manual memory cleanup at regular intervals.

    Supplements torch_empty_cache_steps with more aggressive cleanup
    including Python garbage collection for PIL objects.
    """
    def __init__(self, cleanup_interval=100):
        self.cleanup_interval = cleanup_interval
        self.steps_since_cleanup = 0

    # Provide all callback methods required by transformers trainer
    def on_init_end(self, args, state, control, **kwargs):
        pass

    def on_train_begin(self, args, state, control, **kwargs):
        pass

    def on_train_end(self, args, state, control, **kwargs):
        pass

    def on_epoch_begin(self, args, state, control, **kwargs):
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        pass

    def on_step_begin(self, args, state, control, **kwargs):
        pass

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if not Config.ENABLE_MEMORY_MANAGEMENT:
            return

        self.steps_since_cleanup += 1

        if self.steps_since_cleanup >= self.cleanup_interval:
            cleanup_memory()
            self.steps_since_cleanup = 0

    def on_evaluate(self, args, state, control, **kwargs):
        pass

    def on_predict(self, args, state, control, **kwargs):
        pass

    def on_save(self, args, state, control, **kwargs):
        pass

    def on_log(self, args, state, control, **kwargs):
        pass

    def on_substep_end(self, args, state, control, **kwargs):
        """Called at the end of each substep during gradient accumulation."""
        pass

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Called before optimizer step (after gradient clipping)."""
        pass

    def on_optimizer_step(self, args, state, control, **kwargs):
        """Called after optimizer step (before gradients zeroed)."""
        pass

    def on_prediction_step(self, args, state, control, **kwargs):
        """Called after a prediction step."""
        pass

    def on_push_begin(self, args, state, control, **kwargs):
        """Called before pushing to hub."""
        pass


# ============================================================================
# TOP-K FILTERING (Strategy F)
# ============================================================================

def apply_topk_filtering(
    data: List[Dict],
    config: Dict[str, int]
) -> Tuple[List[Dict], Dict[str, List[str]]]:
    """
    Filter diseases to Top-K per group, map others to "Other_GroupX".

    Args:
        data: List of dicts with 'disease_group' and 'disease' fields
        config: TOP_K_CONFIG dictionary mapping group name to K value

    Returns:
        data: Modified list with 'disease_label_stage2' and 'is_topk' fields
        top_diseases_per_group: Dict mapping group -> list of top K disease names
    """
    # Group data by disease group
    grouped_data = defaultdict(list)
    for item in data:
        grouped_data[item['disease_group']].append(item)

    top_diseases_per_group = {}

    # For each group, find Top-K diseases
    for group_name, group_data in grouped_data.items():
        k = config.get(group_name, 5)  # Default K=5

        # Count disease frequencies within this group
        disease_counts = Counter(item['disease'] for item in group_data)

        # Get top K diseases
        top_k = [disease for disease, _ in disease_counts.most_common(k)]
        top_diseases_per_group[group_name] = top_k

        # Create "Other" label for this group
        group_num = group_name.split('.')[0]
        other_label = f"Other_Group{group_num}"

        # Map diseases to labels
        for item in group_data:
            if item['disease'] in top_k:
                item['disease_label_stage2'] = item['disease']
                item['is_topk'] = True
            else:
                item['disease_label_stage2'] = other_label
                item['is_topk'] = False

    return data, top_diseases_per_group


def display_topk_analysis(data: List[Dict], top_diseases_per_group: Dict, config: Dict):
    """
    Display all diseases with Top-K markers for verification.
    """
    print("\n" + "=" * 80)
    print("DISEASE DISTRIBUTION - TOP-K vs OTHER (Strategy F)")
    print("=" * 80)

    total_topk = 0
    total_other = 0

    for group_name in sorted(config.keys()):
        k = config[group_name]
        group_data = [item for item in data if item['disease_group'] == group_name]
        disease_counts = Counter(item['disease'] for item in group_data)

        print(f"\n{group_name} (Top-{k} selected)")
        print("-" * 80)

        topk_diseases = set(top_diseases_per_group[group_name])
        topk_samples = sum(count for disease, count in disease_counts.items() if disease in topk_diseases)
        other_samples = sum(count for disease, count in disease_counts.items() if disease not in topk_diseases)

        total_topk += topk_samples
        total_other += other_samples

        print(f"  Total: {len(group_data)} samples, {len(disease_counts)} unique diseases")
        print(f"  Top-{k}: {topk_samples} samples ({topk_samples/len(group_data)*100:.1f}%)")
        print(f"  Other: {other_samples} samples ({other_samples/len(group_data)*100:.1f}%)")
        print()

        for rank, (disease, count) in enumerate(disease_counts.most_common(), 1):
            marker = "  [TOP-K]" if disease in topk_diseases else "  [Other]"
            print(f"  {rank:3d}. {disease:<50s} {count:4d}{marker}")

    print("\n" + "=" * 80)
    print(f"OVERALL SUMMARY:")
    print(f"  Total Top-K samples: {total_topk} ({total_topk/(total_topk+total_other)*100:.1f}%)")
    print(f"  Total Other samples: {total_other} ({total_other/(total_topk+total_other)*100:.1f}%)")
    n_groups = len(top_diseases_per_group)
    print(f"  Stage 2 classes: {sum(len(v) for v in top_diseases_per_group.values())} diseases + {n_groups} 'Other' = {sum(len(v) for v in top_diseases_per_group.values()) + n_groups}")
    print("=" * 80)


# ============================================================================
# STRATIFIED SPLIT (from train_two_stage_FuzzyTopK.py)
# ============================================================================

def stratified_split(
    data: List[Dict],
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stratified train/val split that maintains class proportions.

    For 3-stage pipeline, we stratify by disease_label_stage2.
    Falls back to disease_group stratification if any class has too few samples.
    """
    labels = [item["disease_label_stage2"] for item in data]
    indices = list(range(len(data)))

    try:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
    except ValueError as e:
        # Fallback: stratify by group (all groups have ≥41 samples)
        print(f"  [WARN] Disease-level stratification failed: {e}")
        print(f"  [WARN] Falling back to group-level stratification")
        group_labels = [item["disease_group"] for item in data]
        train_indices, val_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=group_labels,
        )

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]

    # Verify stratification
    train_counts = Counter(item["disease_label_stage2"] for item in train_data)
    val_counts = Counter(item["disease_label_stage2"] for item in val_data)

    print(f"\nStratified Split (seed={seed}, test_size={test_size}):")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"\n  {'Class':<50s} {'Train':>6s} {'Val':>5s} {'Train%':>7s}")
    print(f"  {'-' * 70}")

    all_classes = sorted(set(labels), key=lambda x: -(train_counts.get(x, 0) + val_counts.get(x, 0)))
    for cls in all_classes[:10]:  # Show top 10 for brevity
        tr = train_counts.get(cls, 0)
        va = val_counts.get(cls, 0)
        ratio = tr / (tr + va) * 100 if (tr + va) > 0 else 0
        print(f"  {cls:<50s} {tr:6d} {va:5d} {ratio:6.1f}%")

    if len(all_classes) > 10:
        print(f"  ... and {len(all_classes) - 10} more classes")

    return train_data, val_data


# ============================================================================
# CLASS BALANCING (from train_two_stage_FuzzyTopK.py)
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
    - "inverse": target = max_count for all classes (full equalization)
    - "none":    no oversampling, return data as-is

    Only apply to training data, never validation.
    """
    if method == "none":
        print("\nOversampling: disabled (method='none')")
        return data

    random.seed(seed)
    class_counts = Counter(item["disease_label_stage2"] for item in data)
    max_count = max(class_counts.values())

    # Compute target counts (capped at 3x to prevent overfitting on duplicates)
    MAX_OVERSAMPLE_RATIO = 3.0

    if method == "sqrt":
        targets = {
            cls: min(
                int(max_count * math.sqrt(count / max_count)),
                int(count * MAX_OVERSAMPLE_RATIO),
            )
            for cls, count in class_counts.items()
        }
    elif method == "inverse":
        targets = {
            cls: min(max_count, int(count * MAX_OVERSAMPLE_RATIO))
            for cls, count in class_counts.items()
        }
    else:
        raise ValueError(f"Unknown balance method: {method}")

    # Group data by class
    class_data = defaultdict(list)
    for item in data:
        class_data[item["disease_label_stage2"]].append(item)

    # Oversample
    balanced_data = []
    print(f"\nOversampling (method='{method}'):")
    print(f"  {'Class':<50s} {'Before':>7s} {'After':>7s} {'Action'}")
    print(f"  {'-' * 80}")

    # Get top 10 classes for display (sorted by count, descending)
    top_10_classes = sorted(class_counts, key=lambda x: -class_counts[x])[:10]
    top_10_set = set(top_10_classes)

    # Process top 10 classes with display
    for cls in top_10_classes:
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

    # Process remaining classes without display
    for cls in class_counts:
        if cls not in top_10_set:
            original = class_data[cls]
            target = targets[cls]
            current = len(original)

            if current >= target:
                balanced_data.extend(original)
            else:
                repeated = original.copy()
                while len(repeated) < target:
                    repeated.append(random.choice(original))
                balanced_data.extend(repeated[:target])

    random.shuffle(balanced_data)

    old_min = min(class_counts.values())
    old_max = max(class_counts.values())
    new_counts = Counter(item["disease_label_stage2"] for item in balanced_data)
    new_min = min(new_counts.values())
    new_max = max(new_counts.values())

    print(f"\n  Total: {len(data)} -> {len(balanced_data)}")
    print(f"  Imbalance ratio: {old_max / old_min:.2f}:1 -> {new_max / new_min:.2f}:1")

    return balanced_data


def oversample_groups(
    data: List[Dict],
    method: str = "sqrt",
    seed: int = 42,
) -> List[Dict]:
    """
    Oversample minority groups to reduce group-level imbalance for Stage 1.

    Unlike oversample_data() which balances by disease_label_stage2,
    this function balances by disease_group for better Stage 1 training.

    Methods:
    - "sqrt":    target = max_count * sqrt(count / max_count)
    - "inverse": target = max_count for all groups (full equalization)
    - "none":    no oversampling, return data as-is
    """
    if method == "none":
        print("\nGroup oversampling: disabled (method='none')")
        return data

    random.seed(seed)
    group_counts = Counter(item["disease_group"] for item in data)
    max_count = max(group_counts.values())

    # Cap at 3x to prevent overfitting on duplicates
    MAX_OVERSAMPLE_RATIO = 3.0

    if method == "sqrt":
        targets = {
            grp: min(
                int(max_count * math.sqrt(count / max_count)),
                int(count * MAX_OVERSAMPLE_RATIO),
            )
            for grp, count in group_counts.items()
        }
    elif method == "inverse":
        targets = {
            grp: min(max_count, int(count * MAX_OVERSAMPLE_RATIO))
            for grp, count in group_counts.items()
        }
    else:
        raise ValueError(f"Unknown balance method: {method}")

    group_data = defaultdict(list)
    for item in data:
        group_data[item["disease_group"]].append(item)

    balanced_data = []
    print(f"\nGroup oversampling (method='{method}'):")
    print(f"  {'Group':<45s} {'Before':>7s} {'After':>7s} {'Action'}")
    print(f"  {'-' * 75}")

    for grp in sorted(group_counts.keys()):
        original = group_data[grp]
        target = targets[grp]
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

        print(f"  {grp:<45s} {current:7d} {target:7d}  {action}")

    random.shuffle(balanced_data)

    old_min = min(group_counts.values())
    old_max = max(group_counts.values())
    new_counts = Counter(item["disease_group"] for item in balanced_data)
    new_min = min(new_counts.values())
    new_max = max(new_counts.values())

    print(f"\n  Total: {len(data)} -> {len(balanced_data)}")
    print(f"  Group imbalance ratio: {old_max / old_min:.2f}:1 -> {new_max / new_min:.2f}:1")

    return balanced_data


# ============================================================================
# SPLIT INFO PERSISTENCE
# ============================================================================

def save_split_info_3stage(
    train_data: List[Dict],
    val_data: List[Dict],
    top_diseases_per_group: Dict[str, List[str]],
):
    """Save split metadata for inference reproducibility."""
    info = {
        "top_diseases_per_group": top_diseases_per_group,
        "train_image_paths": [item["image_path"] for item in train_data],
        "val_image_paths": [item["image_path"] for item in val_data],
        "fuzzy_threshold": Config.FUZZY_THRESHOLD,
        "top_k_config": Config.TOP_K_CONFIG,
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
    """Stage 1 & 2: Classification format - loads image lazily from path"""
    image = Image.open(sample["image_path"]).convert("RGB")

    # Apply augmentation (only for training, not validation)
    if sample.get("is_training", False):
        image = apply_image_augmentation(image, apply_aug=True)

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


def convert_caption(sample):
    """Stage 3: Caption format - loads image lazily from path"""
    image = Image.open(sample["image_path"]).convert("RGB")

    # Apply augmentation (only for training, not validation)
    if sample.get("is_training", False):
        image = apply_image_augmentation(image, apply_aug=True)

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


def prepare_group_classification_data(data: List[Dict], num_prompts: int = 3, is_training: bool = True) -> Dataset:
    """Prepare data for Stage 1: Group Classification"""
    print("Preparing group classification dataset...")

    samples = []
    for item in data:
        selected_prompts = random.sample(
            GROUP_CLASSIFICATION_PROMPTS, min(num_prompts, len(GROUP_CLASSIFICATION_PROMPTS))
        )
        for prompt in selected_prompts:
            samples.append({
                "image_path": item["image_path"],
                "prompt": prompt,
                "answer": item['disease_group'],
                "is_training": is_training,
            })

    random.shuffle(samples)
    print(f"Created {len(samples)} group classification samples")
    return Dataset.from_list(samples)


def prepare_disease_classification_data(
    data: List[Dict],
    num_prompts: int = 3,
    is_training: bool = True,
    group_context_field: str = None,
) -> Dataset:
    """Prepare data for Stage 2: Disease Classification

    Args:
        group_context_field: Field to use for group context in prompts.
            None    → M0: plain disease prompts, no group context
            'disease_group'   → M1/M2/M4: GT group label in prompt
            'predicted_group' → M3: Stage 1 predicted group in prompt
    """
    print("Preparing disease classification dataset...")

    samples = []
    for item in data:
        if group_context_field is not None:
            group_name = item.get(group_context_field) or item.get('disease_group', '')
            available = [p.replace('{group}', group_name) for p in DISEASE_PROMPTS_WITH_GROUP]
        else:
            available = DISEASE_CLASSIFICATION_PROMPTS

        selected_prompts = random.sample(available, min(num_prompts, len(available)))
        for prompt in selected_prompts:
            samples.append({
                "image_path": item["image_path"],
                "prompt": prompt,
                "answer": f"This image shows {item['disease_label_stage2']}.",
                "is_training": is_training,
            })

    random.shuffle(samples)
    print(f"Created {len(samples)} disease classification samples")
    return Dataset.from_list(samples)


def prepare_caption_data(data: List[Dict], num_prompts: int = 3, is_training: bool = True) -> Dataset:
    """Prepare data for Stage 3: Caption"""
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
                "is_training": is_training,
            })

    random.shuffle(samples)
    print(f"Created {len(samples)} caption samples")
    return Dataset.from_list(samples)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_base_model():
    """Load fresh base model for Stage 1"""
    from unsloth import FastVisionModel

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


def load_from_merged_model(merged_model_path: str):
    """Load merged model from previous stage and add fresh LoRA adapters

    This is the correct approach for multi-stage training:
    - Load the merged model (base weights + previous stage's learned weights)
    - Add NEW LoRA adapters optimized for the current stage
    - Each stage gets fresh, focused adapters
    """
    from unsloth import FastVisionModel

    print(f"\nLoading merged model from previous stage: {merged_model_path}")

    # Load the merged model (no LoRA adapters, just full weights)
    model, tokenizer = FastVisionModel.from_pretrained(
        merged_model_path,
        load_in_4bit=Config.LOAD_IN_4BIT,
        use_gradient_checkpointing="unsloth",
    )

    # Add FRESH LoRA adapters for this stage
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

    print(f"Model loaded from merged weights + fresh LoRA adapters added")
    print(f"  LoRA config: r={Config.LORA_R}, alpha={Config.LORA_ALPHA}")
    return model, tokenizer


# ============================================================================
# DATA PREPARATION PIPELINE
# ============================================================================

def prepare_data_pipeline_3stage() -> Tuple[List[Dict], List[Dict], Dict[str, List[str]]]:
    """
    Complete data pipeline for 3-stage training.

    Returns:
        train_data: Training samples
        val_data: Validation samples
        top_diseases_per_group: Dict mapping group -> list of top K diseases
    """
    # [1/6] Load data
    print("\n[1/6] Loading data...")
    data = load_skincap_data()

    # [1b] Filter to top-N diseases only (Strategy G)
    before_filter = len(data)
    data = [item for item in data if norm_disease(item["disease"]) in ACTIVE_TOP_DISEASES]
    print(f"  Strategy G filter: {before_filter} -> {len(data)} samples (top-{Config.TOP_N} diseases only)")

    # [2/6] Fuzzy consolidation
    print("\n[2/6] Consolidating disease labels (fuzzy matching)...")
    data = fuzzy_consolidate_diseases(data, threshold=Config.FUZZY_THRESHOLD)

    # [3/6] Group categorization
    print("\n[3/6] Categorizing diseases into morphology groups...")
    for item in data:
        item['disease_group'] = categorize_morphology(item['disease'])

    # Verify group distribution
    group_counts = Counter(item['disease_group'] for item in data)
    print("\nGroup Distribution:")
    for group, count in sorted(group_counts.items()):
        print(f"  {group}: {count} samples")

    # [4/6] Top-K filtering
    print("\n[4/6] Applying Top-K filtering per group (Strategy G)...")
    data, top_diseases_per_group = apply_topk_filtering(data, Config.TOP_K_CONFIG)
    display_topk_analysis(data, top_diseases_per_group, Config.TOP_K_CONFIG)

    # [5/6] Stratified split — reuse existing split if available (locked for comparability)
    print("\n[5/6] Loading or creating train/val split...")

    def _extract_fname(path: str) -> str:
        return path.split("\\")[-1] if "\\" in path else path.split("/")[-1]

    if Path(Config.SPLIT_INFO_PATH).exists():
        print(f"  Reusing existing split from {Config.SPLIT_INFO_PATH}")
        with open(Config.SPLIT_INFO_PATH, 'r', encoding='utf-8') as f:
            split_info = json.load(f)
        train_fnames = {_extract_fname(p) for p in split_info["train_image_paths"]}
        val_fnames   = {_extract_fname(p) for p in split_info["val_image_paths"]}
        train_data = [item for item in data if _extract_fname(item["image_path"]) in train_fnames]
        val_data   = [item for item in data if _extract_fname(item["image_path"]) in val_fnames]
        print(f"  Loaded: {len(train_data)} train, {len(val_data)} val")
        print(f"  (Delete {Config.SPLIT_INFO_PATH} to regenerate the split)")
    else:
        print(f"  No split found — creating new split and saving to {Config.SPLIT_INFO_PATH}")
        train_data, val_data = stratified_split(
            data, test_size=Config.TEST_SIZE, seed=Config.SEED
        )
        # [6/6] Save metadata
        print("\n[6/6] Saving split info and configuration...")
        save_split_info_3stage(train_data, val_data, top_diseases_per_group)

    return train_data, val_data, top_diseases_per_group


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_stage1_group(train_data: List[Dict], val_data: List[Dict]):
    """
    Stage 1: Group Classification Training
    Strategy G: top-15 diseases, 3 or 4 groups (set via Config.GROUP_MODE)
    """
    from unsloth import is_bf16_supported
    from trl import SFTTrainer, SFTConfig

    num_groups = len(Config.TOP_K_CONFIG)
    print("\n" + "=" * 60)
    print(f"STAGE 1: GROUP CLASSIFICATION TRAINING ({num_groups} groups, mode={Config.GROUP_MODE})")
    print("=" * 60)

    # Strategy G: top-15 dataset is already balanced enough (≤5:1 ratio)
    # No oversampling needed — use training data directly
    balanced_train = train_data

    # Load base model
    model, tokenizer = load_base_model()

    # Prepare data
    train_dataset = prepare_group_classification_data(balanced_train, num_prompts=3, is_training=True)
    val_dataset = prepare_group_classification_data(val_data, num_prompts=1, is_training=False)

    # Convert format
    num_proc = calculate_dataset_num_proc(Config.DATASET_NUM_PROC)
    train_dataset = train_dataset.map(
        convert_classification,
        remove_columns=["image_path", "prompt", "answer", "is_training"],
        desc="Formatting train data",
        num_proc=num_proc,
    )
    val_dataset = val_dataset.map(
        convert_classification,
        remove_columns=["image_path", "prompt", "answer", "is_training"],
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
        eval_strategy="epoch",
        report_to="none",
        torch_empty_cache_steps=Config.TORCH_EMPTY_CACHE_STEPS,
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
        callbacks=[MemoryCleanupCallback(cleanup_interval=Config.MANUAL_CLEANUP_INTERVAL)],
    )

    # Train
    print("\nTraining Stage 1...")
    trainer.train()

    # Save Model 1 (Group Classification)
    print("\n" + "=" * 60)
    print("SAVING MODEL 1: GROUP CLASSIFICATION (10 groups)")
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


def generate_stage1_predictions_on_train(train_data: List[Dict]) -> List[Dict]:
    """
    Run Stage 1 on all training images and attach predicted group labels.
    Used by M3: Stage 2 trains with noisy (Stage 1 predicted) group context.
    Caches predictions to STAGE1_TRAIN_PREDICTIONS_PATH for reproducibility.
    """
    import torch
    import gc

    predictions_path = Config.STAGE1_TRAIN_PREDICTIONS_PATH

    # Use cached predictions if already computed
    if Path(predictions_path).exists():
        print(f"  [M3] Loading cached Stage 1 train predictions from {predictions_path}")
        with open(predictions_path, 'r', encoding='utf-8') as f:
            pred_map = json.load(f)
        for item in train_data:
            item['predicted_group'] = pred_map.get(item['image_path'], item['disease_group'])
        matched = sum(1 for item in train_data if item['predicted_group'] == item['disease_group'])
        print(f"  [M3] Train-set Stage 1 accuracy (cached): {matched}/{len(train_data)} "
              f"({matched/len(train_data)*100:.1f}%)")
        return train_data

    print("\n[M3] Running Stage 1 inference on training set to get predicted group labels...")
    from inference_group_classification import GroupClassificationInference
    stage1 = GroupClassificationInference(Config.OUTPUT_DIR_STAGE1_MERGED)
    results = stage1.predict_batch(train_data, batch_size=8)

    pred_map = {}
    for item, result in zip(train_data, results):
        predicted = result.get('predicted') or item['disease_group']
        item['predicted_group'] = predicted
        pred_map[item['image_path']] = predicted

    # Save for reproducibility
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(pred_map, f, indent=2, ensure_ascii=False)
    print(f"  [M3] Stage 1 train predictions saved to {predictions_path}")

    matched = sum(1 for item in train_data if item['predicted_group'] == item['disease_group'])
    print(f"  [M3] Train-set Stage 1 accuracy: {matched}/{len(train_data)} "
          f"({matched/len(train_data)*100:.1f}%)")

    # Free GPU memory before Stage 2 training
    del stage1
    torch.cuda.empty_cache()
    gc.collect()

    return train_data


def train_stage2_disease(train_data: List[Dict], val_data: List[Dict]):
    """
    Stage 2: Disease Classification Training (Top-K diseases + 5 "Other")

    Loads from Stage 1 checkpoint.
    """
    from unsloth import is_bf16_supported
    from trl import SFTTrainer, SFTConfig

    print("\n" + "=" * 60)
    print("STAGE 2: DISEASE CLASSIFICATION TRAINING")
    print("=" * 60)

    # Optionally replace train split with fuzzytopk train images (SCCIS class)
    if Config.USE_FUZZYTOPK_SPLIT_FOR_STAGE2:
        fk_split_path = Path(__file__).parent / "split_info_fuzzytopk.json"
        with open(fk_split_path, 'r', encoding='utf-8') as f:
            fk_split = json.load(f)
        def _extract_fname(p: str) -> str:
            return p.split("\\")[-1] if "\\" in p else p.split("/")[-1]
        fk_train_fnames = {_extract_fname(p) for p in fk_split["train_image_paths"]}
        train_data = [item for item in train_data if _extract_fname(item["image_path"]) in fk_train_fnames]
        print(f"  [fuzzytopk_split] Filtered to fuzzytopk train images: {len(train_data)} samples")
        print(f"  [fuzzytopk_split] Val stays as 3stage: {len(val_data)} samples")

    # Balance training data (NOT validation)
    balanced_train = oversample_data(
        train_data, method=Config.BALANCE_METHOD, seed=Config.SEED
    )

    # Determine group context field based on Stage 2 method
    method = Config.STAGE2_METHOD
    if method == "M0":
        group_context_field = None
    elif method in ("M1", "M2", "M4"):
        group_context_field = 'disease_group'        # GT group labels
    elif method == "M3":
        balanced_train = generate_stage1_predictions_on_train(balanced_train)
        group_context_field = 'predicted_group'      # Stage 1 predicted labels
    else:
        group_context_field = None

    print(f"\n  Stage 2 method: {method}, group context field: {group_context_field or 'none'}")

    # Load model — base model avoids cascade penalty from Stage 1 weights
    if Config.USE_BASE_MODEL_FOR_STAGE2:
        print("  [base model] Loading from BASE model (no cascade penalty)")
        model, tokenizer = load_base_model()
    else:
        model, tokenizer = load_from_merged_model(Config.OUTPUT_DIR_STAGE1_MERGED)

    # Prepare data
    train_dataset = prepare_disease_classification_data(
        balanced_train, num_prompts=3, is_training=True,
        group_context_field=group_context_field,
    )
    val_dataset = prepare_disease_classification_data(
        val_data, num_prompts=1, is_training=False,
        group_context_field=None,  # No group context in training-time val eval
    )

    # Convert format
    num_proc = calculate_dataset_num_proc(Config.DATASET_NUM_PROC)
    train_dataset = train_dataset.map(
        convert_classification,
        remove_columns=["image_path", "prompt", "answer", "is_training"],
        desc="Formatting train data",
        num_proc=num_proc,
    )
    val_dataset = val_dataset.map(
        convert_classification,
        remove_columns=["image_path", "prompt", "answer", "is_training"],
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
        eval_strategy="epoch",
        report_to="none",
        torch_empty_cache_steps=Config.TORCH_EMPTY_CACHE_STEPS,
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
        callbacks=[MemoryCleanupCallback(cleanup_interval=Config.MANUAL_CLEANUP_INTERVAL)],
    )

    # Train
    print("\nTraining Stage 2...")
    trainer.train()

    # Save Model 2 (Disease Classification)
    print("\n" + "=" * 60)
    print("SAVING MODEL 2: DISEASE CLASSIFICATION")
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


def train_stage3_caption(train_data: List[Dict], val_data: List[Dict]):
    """
    Stage 3: Caption Generation Training

    Loads from Stage 2 checkpoint.
    """
    from unsloth import is_bf16_supported
    from trl import SFTTrainer, SFTConfig

    print("\n" + "=" * 60)
    print("STAGE 3: CAPTION GENERATION TRAINING")
    print("=" * 60)

    # Balance training data (NOT validation)
    balanced_train = oversample_data(
        train_data, method=Config.BALANCE_METHOD, seed=Config.SEED
    )

    # Load from source model — default is Stage 2 merged, but can be overridden
    # via Config.STAGE3_SOURCE_MODEL (e.g. to start from fuzzytopk_s1cascade)
    source = Config.STAGE3_SOURCE_MODEL if Config.STAGE3_SOURCE_MODEL else Config.OUTPUT_DIR_STAGE2_MERGED
    print(f"  Stage 3 source model: {source}")
    model, tokenizer = load_from_merged_model(source)

    # Prepare data
    train_dataset = prepare_caption_data(balanced_train, num_prompts=3, is_training=True)
    val_dataset = prepare_caption_data(val_data, num_prompts=1, is_training=False)

    # Convert format
    num_proc = calculate_dataset_num_proc(Config.DATASET_NUM_PROC)
    train_dataset = train_dataset.map(
        convert_caption,
        remove_columns=["image_path", "prompt", "caption", "is_training"],
        desc="Formatting train data",
        num_proc=num_proc,
    )
    val_dataset = val_dataset.map(
        convert_caption,
        remove_columns=["image_path", "prompt", "caption", "is_training"],
        desc="Formatting val data",
        num_proc=num_proc,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Training args
    training_args = SFTConfig(
        output_dir=Config.OUTPUT_DIR_STAGE3,
        num_train_epochs=Config.STAGE3_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=Config.WARMUP_STEPS,
        learning_rate=Config.STAGE3_LR if Config.STAGE3_LR > 0 else Config.LEARNING_RATE,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=Config.WEIGHT_DECAY,
        lr_scheduler_type="linear",
        seed=Config.SEED,
        save_strategy="epoch",
        report_to="none",
        torch_empty_cache_steps=Config.TORCH_EMPTY_CACHE_STEPS,
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
        callbacks=[MemoryCleanupCallback(cleanup_interval=Config.MANUAL_CLEANUP_INTERVAL)],
    )

    lr_used = Config.STAGE3_LR if Config.STAGE3_LR > 0 else Config.LEARNING_RATE
    print(f"\nTraining Stage 3 (LR={lr_used:.2e})...")
    trainer.train()

    # Save Model 3 (Caption Generation - FINAL MODEL)
    print("\n" + "=" * 60)
    print("SAVING MODEL 3: CAPTION GENERATION (FINAL MODEL)")
    print("=" * 60)

    model.save_pretrained(Config.OUTPUT_DIR_STAGE3)
    tokenizer.save_pretrained(Config.OUTPUT_DIR_STAGE3)
    print(f"LoRA saved to: {Config.OUTPUT_DIR_STAGE3}")

    model.save_pretrained_merged(
        Config.OUTPUT_DIR_STAGE3_MERGED,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved to: {Config.OUTPUT_DIR_STAGE3_MERGED}")

    return model, tokenizer


# ============================================================================
# MAIN
# ============================================================================

def train_three_stage():
    """Full three-stage training pipeline with Strategy F (Expert Dictionary 10-Group)"""

    print("=" * 60)
    print("THREE-STAGE TRAINING PIPELINE (Strategy F - Expert Dictionary 10-Group)")
    print("=" * 60)
    print(f"Model:            {Config.MODEL_NAME}")
    print(f"Seed:             {Config.SEED}")
    print(f"Stage 1 epochs:   {Config.STAGE1_EPOCHS}")
    print(f"Stage 2 epochs:   {Config.STAGE2_EPOCHS}")
    print(f"Stage 3 epochs:   {Config.STAGE3_EPOCHS}")
    print(f"Fuzzy threshold:  {Config.FUZZY_THRESHOLD}")
    print(f"Balance method:   {Config.BALANCE_METHOD}")
    print("=" * 60)

    setup_environment()

    # Data pipeline
    train_data, val_data, top_diseases = prepare_data_pipeline_3stage()
    print(f"\nTrain: {len(train_data)}, Val: {len(val_data)}")

    # Stage 1: Group Classification
    print("\n[STAGE 1/3] Group Classification...")
    train_stage1_group(train_data, val_data)

    # Stage 2: Disease Classification (loads from Stage 1)
    print("\n[STAGE 2/3] Disease Classification...")
    train_stage2_disease(train_data, val_data)

    # Stage 3: Caption Generation (loads from Stage 2)
    print("\n[STAGE 3/3] Caption Generation...")
    train_stage3_caption(train_data, val_data)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"""
OUTPUT MODELS (Strategy F - Expert Dictionary 10-Group):

  MODEL 1: GROUP CLASSIFICATION (10 classes)
    LoRA:   {Config.OUTPUT_DIR_STAGE1}
    Merged: {Config.OUTPUT_DIR_STAGE1_MERGED}

  MODEL 2: DISEASE CLASSIFICATION (Top-K diseases + 10 'Other')
    LoRA:   {Config.OUTPUT_DIR_STAGE2}
    Merged: {Config.OUTPUT_DIR_STAGE2_MERGED}

  MODEL 3: CAPTION GENERATION (FINAL MODEL)
    LoRA:   {Config.OUTPUT_DIR_STAGE3}
    Merged: {Config.OUTPUT_DIR_STAGE3_MERGED}

  Split info: {Config.SPLIT_INFO_PATH}
""")


def train_stage1_only():
    """Train only Stage 1 (Group Classification)"""
    setup_environment()
    train_data, val_data, top_diseases = prepare_data_pipeline_3stage()
    train_stage1_group(train_data, val_data)


def train_stage2_only():
    """Train only Stage 2 (requires Stage 1 checkpoint)"""
    if not Path(Config.OUTPUT_DIR_STAGE1).exists():
        print(f"Error: Stage 1 checkpoint not found at {Config.OUTPUT_DIR_STAGE1}")
        print("Please run Stage 1 first.")
        return

    setup_environment()
    train_data, val_data, top_diseases = prepare_data_pipeline_3stage()
    train_stage2_disease(train_data, val_data)


def train_stage3_only():
    """Train only Stage 3 (requires Stage 2 checkpoint or a custom --stage3_source model)"""
    source = Config.STAGE3_SOURCE_MODEL if Config.STAGE3_SOURCE_MODEL else Config.OUTPUT_DIR_STAGE2_MERGED
    if not Path(source).exists():
        print(f"Error: Stage 3 source model not found at {source}")
        if Config.STAGE3_SOURCE_MODEL:
            print("Check the --stage3_source path.")
        else:
            print("Please run Stage 2 first, or specify --stage3_source.")
        return

    setup_environment()
    train_data, val_data, top_diseases = prepare_data_pipeline_3stage()
    train_stage3_caption(train_data, val_data)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Three-Stage Training with Strategy F (Expert Dictionary 10-Group Top-K)"
    )
    parser.add_argument("--mode", type=str, default="both",
                        choices=["both", "stage1", "stage2", "stage3"],
                        help="Training mode")
    parser.add_argument("--stage1_epochs", type=int, default=Config.STAGE1_EPOCHS)
    parser.add_argument("--stage2_epochs", type=int, default=Config.STAGE2_EPOCHS)
    parser.add_argument("--stage3_epochs", type=int, default=Config.STAGE3_EPOCHS)
    parser.add_argument("--stage2_method", type=str, default=Config.STAGE2_METHOD,
                        choices=["M0", "M1", "M2", "M3", "M4"],
                        help="Stage 2 training method (M0=no group context, M1=GT group, "
                             "M2=GT train/predicted infer, M3=predicted train+infer, M4=GT train/soft infer)")
    parser.add_argument("--balance", type=str, default="sqrt",
                        choices=["sqrt", "inverse", "none"],
                        help="Class balancing method")
    parser.add_argument("--fuzzy_threshold", type=int, default=91,
                        help="Fuzzy matching threshold (0-100)")
    parser.add_argument("--dataset_num_proc", type=int, default=None,
                        help="Number of processes for dataset formatting (default: auto)")
    parser.add_argument("--use_base_model", action="store_true", default=False,
                        help="Load base model for Stage 2 instead of Stage 1 merged (avoids cascade penalty)")
    parser.add_argument("--use_fuzzytopk_split", action="store_true", default=False,
                        help="Use split_info_fuzzytopk.json train images for Stage 2 (val stays as 3stage for fair comparison)")
    parser.add_argument("--stage3_source", type=str, default="",
                        help="Source model path for Stage 3 training (default: Stage 2 merged). "
                             "E.g. './skincap_fuzzytopk_s1cascade_classification_merged'")
    parser.add_argument("--stage3_lr", type=float, default=0.0,
                        help="Learning rate for Stage 3 (default: same as global LR=2e-4). "
                             "Use lower value e.g. 5e-5 to prevent catastrophic forgetting.")

    args = parser.parse_args()

    Config.STAGE1_EPOCHS = args.stage1_epochs
    Config.STAGE2_EPOCHS = args.stage2_epochs
    Config.STAGE3_EPOCHS = args.stage3_epochs
    Config.STAGE2_METHOD = args.stage2_method
    Config.USE_BASE_MODEL_FOR_STAGE2 = args.use_base_model
    if Config.USE_BASE_MODEL_FOR_STAGE2:
        Config.OUTPUT_DIR_STAGE2 = f"./skincap_3stage_disease_{Config.STAGE2_METHOD}_base"
        Config.OUTPUT_DIR_STAGE2_MERGED = f"./skincap_3stage_disease_{Config.STAGE2_METHOD}_base_merged"
    else:
        Config.OUTPUT_DIR_STAGE2 = f"./skincap_3stage_disease_{Config.STAGE2_METHOD}"
        Config.OUTPUT_DIR_STAGE2_MERGED = f"./skincap_3stage_disease_{Config.STAGE2_METHOD}_merged"
    Config.BALANCE_METHOD = args.balance
    Config.FUZZY_THRESHOLD = args.fuzzy_threshold
    Config.DATASET_NUM_PROC = args.dataset_num_proc
    Config.USE_FUZZYTOPK_SPLIT_FOR_STAGE2 = args.use_fuzzytopk_split
    if Config.USE_FUZZYTOPK_SPLIT_FOR_STAGE2:
        base_suffix = "_base" if Config.USE_BASE_MODEL_FOR_STAGE2 else ""
        Config.OUTPUT_DIR_STAGE2 = f"./skincap_3stage_disease_{Config.STAGE2_METHOD}{base_suffix}_fuzzytopk"
        Config.OUTPUT_DIR_STAGE2_MERGED = f"./skincap_3stage_disease_{Config.STAGE2_METHOD}{base_suffix}_fuzzytopk_merged"

    Config.STAGE3_SOURCE_MODEL = args.stage3_source
    Config.STAGE3_LR = args.stage3_lr
    if args.stage3_source:
        # Derive output dir name from the source model directory name.
        # E.g. "skincap_fuzzytopk_s1cascade_classification_merged"
        #   → suffix = "fuzzytopk_s1cascade_classification"
        #   → output = "./skincap_stage3_caption_fuzzytopk_s1cascade_classification[_merged]"
        src_name = Path(args.stage3_source).name
        suffix = src_name.replace("_merged", "").replace("skincap_", "")
        Config.OUTPUT_DIR_STAGE3 = f"./skincap_stage3_caption_{suffix}"
        Config.OUTPUT_DIR_STAGE3_MERGED = f"./skincap_stage3_caption_{suffix}_merged"
        print(f"Stage 3 custom source: {args.stage3_source}")
        print(f"Stage 3 output dir:    {Config.OUTPUT_DIR_STAGE3_MERGED}")

    if args.mode == "both":
        train_three_stage()
    elif args.mode == "stage1":
        train_stage1_only()
    elif args.mode == "stage2":
        train_stage2_only()
    elif args.mode == "stage3":
        train_stage3_only()
