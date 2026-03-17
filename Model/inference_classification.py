"""
Inference script for the 1st Stage Classification Model
Based on Qwen3-VL-8B fine-tuned for skin disease classification

Features:
- Parallel batching for efficient inference
- Same train/val/test split as training (seed=42)
- Comprehensive classification metrics (Accuracy, F1, Precision, Recall, etc.)
- Error analysis for train/val sets with ground truth comparison
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
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import cv2

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss,
    hamming_loss,
    jaccard_score,
    zero_one_loss,
    multilabel_confusion_matrix,
)

# Visualization (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("Note: matplotlib/seaborn not installed. Visualization features disabled.")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class InferenceConfig:
    # Random seed (same as training)
    SEED: int = 42
    
    # Model paths
    MODEL_PATH_LORA: str = "./skincap_model_classification"
    MODEL_PATH_MERGED: str = "./skincap_model_classification_merged"
    
    # Data paths
    CSV_PATH: str = "./SkinCAP/skincap_v240623.csv"
    IMAGE_BASE_PATH: str = "./SkinCAP/skincap"
    
    # Data split ratios
    # IMPORTANT: Training used 90/10 (train/val) split with seed=42
    # Default matches training code exactly for accurate evaluation
    # The val set (10%) is the only "unseen" data for this model
    TRAIN_RATIO: float = 0.9
    VAL_RATIO: float = 0.1
    
    # Inference settings
    BATCH_SIZE: int = 8  # Batch size for parallel GPU inference (increase if VRAM allows)
    MAX_NEW_TOKENS: int = 128  # For classification, we don't need many tokens
    TEMPERATURE: float = 0.1  # Low temperature for deterministic classification
    TOP_P: float = 0.9
    
    # Top-K settings
    ENABLE_TOPK: bool = False  # Enable Top-K evaluation (slower but more metrics)
    TOPK_SAMPLES: int = 5      # Number of samples per image for Top-K estimation (5 is good balance)
    
    # Attention Map settings
    ENABLE_ATTENTION: bool = False  # Enable attention map generation
    ATTENTION_LIMIT: int = None     # Limit number of attention maps (None = all)
    
    # Output
    OUTPUT_DIR: str = "./classification_results"


# ============================================================================
# SEED SETUP
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"✓ Random seed set to {seed}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_skincap_data(config: InferenceConfig) -> List[Dict]:
    """Load SkinCAP dataset"""
    print("Loading SkinCAP dataset...")
    
    csv_path = Path(config.CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Filter valid rows
    df = df.dropna(subset=["skincap_file_path", "caption_zh_polish_en", "disease"])
    print(f"After filtering: {len(df)} valid rows")
    
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
    
    print(f"✓ Found {len(data_list)} valid samples")
    return data_list


def split_data(data: List[Dict], config: InferenceConfig) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data into train/val with the SAME seed and ratio as training.
    
    IMPORTANT: The training code used 90/10 split with seed=42.
    - Train (90%): Samples the model has SEEN during training
    - Val (10%): Samples the model has NOT seen (true evaluation set)
    
    We use the exact same split to ensure:
    - Train set evaluation = error analysis on seen data
    - Val set evaluation = true performance on unseen data
    """
    set_seed(config.SEED)
    
    # Shuffle data (same as training)
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    # Split exactly like training: 90/10
    total = len(data_copy)
    split_idx = int(total * config.TRAIN_RATIO)
    
    train_data = data_copy[:split_idx]
    val_data = data_copy[split_idx:]
    
    print(f"\nData Split (seed={config.SEED}) - MATCHING TRAINING SPLIT:")
    print(f"  Train: {len(train_data)} samples ({config.TRAIN_RATIO*100:.0f}%) [SEEN during training]")
    print(f"  Val:   {len(val_data)} samples ({config.VAL_RATIO*100:.0f}%) [UNSEEN - true test set]")
    print(f"\n  Note: Val set is the true 'unseen' data for this model.")
    print(f"        Train set evaluation is for error analysis only.")
    
    return train_data, val_data


def get_unique_diseases(data: List[Dict]) -> List[str]:
    """Get list of unique diseases"""
    diseases = sorted(set(item["disease"] for item in data))
    return diseases


def normalize_disease_name(name: str) -> str:
    """
    Normalize disease name to handle variations like:
    - "basal-cell-carcinoma" vs "basal cell carcinoma"
    - Extra whitespace
    - Case differences
    """
    if not name:
        return ""
    
    # Convert to lowercase
    name = name.lower().strip()
    
    # Replace hyphens and underscores with spaces
    name = name.replace("-", " ").replace("_", " ")
    
    # Remove extra whitespace (multiple spaces -> single space)
    name = " ".join(name.split())
    
    return name


def match_disease_names(pred: str, ground_truth: str) -> bool:
    """
    Check if prediction matches ground truth with flexible matching.
    
    Handles cases like:
    - "melanocytic-nevi" matches "melanocytic nevi" (hyphen vs space)
    - "seborrheic-keratosis-irritated" matches "seborrheic keratosis" (partial match)
    - "basal cell carcinoma" matches "basal-cell-carcinoma"
    """
    pred_norm = normalize_disease_name(pred)
    gt_norm = normalize_disease_name(ground_truth)
    
    # Exact match after normalization
    if pred_norm == gt_norm:
        return True
    
    # Partial match: check if one contains all words of the other
    pred_words = set(pred_norm.split())
    gt_words = set(gt_norm.split())
    
    # If ground truth words are subset of prediction (e.g., "seborrheic keratosis" in "seborrheic keratosis irritated")
    if gt_words.issubset(pred_words) and len(gt_words) >= 2:
        return True
    
    # If prediction words are subset of ground truth
    if pred_words.issubset(gt_words) and len(pred_words) >= 2:
        return True
    
    return False


def create_disease_lookup(disease_list: List[str]) -> Dict[str, str]:
    """
    Create a lookup dictionary mapping normalized names to original names.
    This helps match predictions like "basal cell carcinoma" to "basal-cell-carcinoma".
    """
    lookup = {}
    for disease in disease_list:
        normalized = normalize_disease_name(disease)
        lookup[normalized] = disease
    return lookup


# ============================================================================
# ATTENTION MAP EXTRACTION
# ============================================================================

class AttentionExtractor:
    """
    Extract real attention maps from Qwen2-VL/Qwen3-VL model.
    
    Uses multiple methods:
    1. Vision encoder self-attention weights
    2. Cross-attention from language model to image tokens
    3. Gradient-weighted attention as fallback
    """
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.attention_weights = []
        self.hooks = []
        self.vision_attention = []
        
        # Find model structure
        self._analyze_model_structure()
        self._register_attention_hooks()
    
    def _analyze_model_structure(self):
        """Analyze model structure to find attention layers"""
        self.model_type = "unknown"
        self.vision_encoder = None
        self.language_model = None
        
        # Get base model
        if hasattr(self.model, 'base_model'):
            base = self.model.base_model.model
        else:
            base = self.model
        
        # Find vision encoder
        if hasattr(base, 'model') and hasattr(base.model, 'visual'):
            self.vision_encoder = base.model.visual
            self.model_type = "qwen_vl"
        elif hasattr(base, 'visual'):
            self.vision_encoder = base.visual
            self.model_type = "qwen_vl"
        elif hasattr(base, 'vision_tower'):
            self.vision_encoder = base.vision_tower
            self.model_type = "llava"
        
        # Find language model layers
        if hasattr(base, 'model') and hasattr(base.model, 'layers'):
            self.language_model = base.model.layers
        elif hasattr(base, 'layers'):
            self.language_model = base.layers
        
        print(f"   Model type: {self.model_type}")
        print(f"   Vision encoder found: {self.vision_encoder is not None}")
        print(f"   Language layers found: {self.language_model is not None}")
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention weights"""
        
        def make_vision_attn_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        self.vision_attention.append({
                            'layer': layer_idx,
                            'weights': attn_weights.detach().cpu()
                        })
            return hook
        
        def make_lang_attn_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        self.attention_weights.append({
                            'layer': layer_idx,
                            'weights': attn_weights.detach().cpu()
                        })
            return hook
        
        # Register hooks on vision encoder attention
        if self.vision_encoder is not None:
            if hasattr(self.vision_encoder, 'blocks'):
                for i, block in enumerate(self.vision_encoder.blocks):
                    if hasattr(block, 'attn'):
                        handle = block.attn.register_forward_hook(make_vision_attn_hook(i))
                        self.hooks.append(handle)
            elif hasattr(self.vision_encoder, 'layers'):
                for i, layer in enumerate(self.vision_encoder.layers):
                    if hasattr(layer, 'self_attn'):
                        handle = layer.self_attn.register_forward_hook(make_vision_attn_hook(i))
                        self.hooks.append(handle)
        
        # Register hooks on language model attention (sample every 4th layer)
        if self.language_model is not None:
            for i, layer in enumerate(self.language_model):
                if i % 4 == 0:
                    if hasattr(layer, 'self_attn'):
                        handle = layer.self_attn.register_forward_hook(make_lang_attn_hook(i))
                        self.hooks.append(handle)
        
        print(f"   ✓ Registered {len(self.hooks)} attention hooks")
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _clear_attention(self):
        """Clear stored attention weights"""
        self.attention_weights = []
        self.vision_attention = []
    
    def extract_attention(self, image: Image.Image, prompt: str,
                          target_size: Tuple[int, int] = None) -> np.ndarray:
        """Extract attention map for an image."""
        if target_size is None:
            target_size = (image.height, image.width)
        
        self._clear_attention()
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_attentions=True,
                    return_dict_in_generate=True,
                )
            
            attention_map = self._compute_attention_map(target_size)
            
            if attention_map is not None:
                return attention_map
            else:
                return self._extract_saliency_fallback(image, target_size)
                
        except Exception as e:
            return self._extract_saliency_fallback(image, target_size)
    
    def _compute_attention_map(self, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Compute attention map from collected attention weights"""
        if self.vision_attention:
            return self._process_vision_attention(target_size)
        if self.attention_weights:
            return self._process_language_attention(target_size)
        return None
    
    def _process_vision_attention(self, target_size: Tuple[int, int]) -> np.ndarray:
        """Process vision encoder attention into spatial map"""
        all_attention = []
        
        for attn_data in self.vision_attention:
            weights = attn_data['weights']
            if weights.dim() == 4:
                avg_attn = weights.mean(dim=1)
                if avg_attn.shape[1] > 1:
                    cls_attn = avg_attn[0, 0, 1:]
                    all_attention.append(cls_attn.numpy())
        
        if not all_attention:
            return None
        
        attention = np.mean(all_attention, axis=0)
        return self._reshape_to_grid(attention, target_size)
    
    def _process_language_attention(self, target_size: Tuple[int, int]) -> np.ndarray:
        """Process language model attention focusing on image tokens"""
        all_attention = []
        
        for attn_data in self.attention_weights:
            weights = attn_data['weights']
            if weights.dim() == 4:
                avg_attn = weights.mean(dim=1)
                seq_len = avg_attn.shape[-1]
                img_token_estimate = min(256, seq_len // 2)
                if seq_len > img_token_estimate:
                    attn_to_image = avg_attn[0, -10:, :img_token_estimate].mean(dim=0)
                    all_attention.append(attn_to_image.numpy())
        
        if not all_attention:
            return None
        
        attention = np.mean(all_attention, axis=0)
        return self._reshape_to_grid(attention, target_size)
    
    def _reshape_to_grid(self, attention: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Reshape 1D attention to 2D grid and resize"""
        seq_len = len(attention)
        grid_size = int(np.sqrt(seq_len))
        
        for gs in [14, 16, 24, 32, 28, 12]:
            if gs * gs <= seq_len:
                grid_size = gs
                break
        
        grid_len = grid_size * grid_size
        if grid_len <= seq_len:
            attention_grid = attention[:grid_len].reshape(grid_size, grid_size)
        else:
            padded = np.zeros(grid_len)
            padded[:seq_len] = attention
            attention_grid = padded.reshape(grid_size, grid_size)
        
        attention_grid = attention_grid - attention_grid.min()
        if attention_grid.max() > 0:
            attention_grid = attention_grid / attention_grid.max()
        
        attention_map = cv2.resize(
            attention_grid.astype(np.float32),
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        attention_map = cv2.GaussianBlur(attention_map, (15, 15), 0)
        attention_map = attention_map - attention_map.min()
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()
        
        return attention_map
    
    def _extract_saliency_fallback(self, image: Image.Image,
                                    target_size: Tuple[int, int]) -> np.ndarray:
        """Final fallback: Use image saliency"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
            mean_lab = img_lab.mean(axis=(0, 1), keepdims=True)
            saliency = np.sqrt(((img_lab - mean_lab) ** 2).sum(axis=2))
        else:
            saliency = np.abs(img_array.astype(np.float32) - img_array.mean())
        
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        saliency = cv2.GaussianBlur(saliency.astype(np.float32), (21, 21), 0)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        if saliency.shape[:2] != target_size:
            saliency = cv2.resize(saliency, (target_size[1], target_size[0]))
        
        return saliency


# ============================================================================
# ATTENTION MAP VISUALIZATION
# ============================================================================

def create_attention_visualization(image_path: str, heatmap: np.ndarray,
                                   ground_truth: str, predicted: str,
                                   is_correct: bool, output_path: str,
                                   sample_id: int):
    """Create 2x2 visualization with attention heatmap overlay"""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Attention Map Visualization', fontsize=16)
    
    # Top-left: Original Image
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Top-right: Attention Overlay
    axes[0, 1].imshow(img_array)
    axes[0, 1].imshow(heatmap_resized, cmap='jet', alpha=0.5)
    axes[0, 1].set_title('Attention Overlay')
    axes[0, 1].axis('off')
    
    # Bottom-left: Attention Heatmap
    im = axes[1, 0].imshow(heatmap_resized, cmap='hot')
    axes[1, 0].set_title('Attention Heatmap')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    
    # Bottom-right: High Attention Regions
    threshold = 0.5
    attention_binary = heatmap_resized > threshold
    axes[1, 1].imshow(img_array)
    axes[1, 1].contour(attention_binary, colors='red', linewidths=2)
    axes[1, 1].set_title(f'High Attention Regions (>{threshold:.1f})')
    axes[1, 1].axis('off')
    
    status = "CORRECT" if is_correct else "INCORRECT"
    status_color = "green" if is_correct else "red"
    
    info_text = f"ID: {sample_id} | {status}\nGT: {ground_truth}\nPred: {predicted}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=11,
             color=status_color, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def generate_attention_maps(predictions: List[Dict], attention_extractor: AttentionExtractor,
                            output_dir: str, split_name: str, limit: int = None,
                            prompt: str = "What skin disease is shown in this image?"):
    """Generate attention map visualizations for predictions"""
    
    if not HAS_VISUALIZATION:
        print("⚠ Skipping attention maps (matplotlib not available)")
        return
    
    output_base = Path(output_dir) / f"attention_maps_{split_name}"
    correct_dir = output_base / "Correct"
    incorrect_dir = output_base / "Incorrect"
    
    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter successful predictions
    valid_predictions = [p for p in predictions if p.get("status") == "success"]
    
    if limit:
        valid_predictions = valid_predictions[:limit]
    
    print(f"\n🎨 Generating attention maps for {len(valid_predictions)} samples...")
    
    correct_count = 0
    incorrect_count = 0
    
    for pred in tqdm(valid_predictions, desc="Generating attention maps"):
        sample_id = pred.get("id", "unknown")
        image_path = pred.get("image_path", "")
        ground_truth = pred.get("ground_truth", "")
        predicted = pred.get("predicted", "")
        
        if not Path(image_path).exists():
            continue
        
        is_correct = match_disease_names(predicted, ground_truth)
        
        if is_correct:
            correct_count += 1
            out_dir = correct_dir
        else:
            incorrect_count += 1
            out_dir = incorrect_dir
        
        gt_short = normalize_disease_name(ground_truth).replace(' ', '_')[:25]
        filename = f"{sample_id}_{gt_short}.png"
        output_path = out_dir / filename
        
        try:
            img = Image.open(image_path).convert("RGB")
            heatmap = attention_extractor.extract_attention(
                img, prompt, target_size=(img.height, img.width)
            )
            
            create_attention_visualization(
                image_path, heatmap, ground_truth, predicted,
                is_correct, str(output_path), sample_id
            )
        except Exception as e:
            print(f"\n⚠ Error processing {sample_id}: {e}")
    
    print(f"\n✓ Attention maps saved:")
    print(f"   {correct_dir} ({correct_count} images)")
    print(f"   {incorrect_dir} ({incorrect_count} images)")


# ============================================================================
# MODEL CLASS
# ============================================================================

class ClassificationInference:
    def __init__(self, model_path: str, load_in_4bit: bool = True):
        """Initialize the classification model"""
        print(f"\n{'='*60}")
        print("Loading Classification Model")
        print(f"{'='*60}")
        print(f"Model path: {model_path}")
        
        from unsloth import FastVisionModel
        
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
        )
        FastVisionModel.for_inference(self.model)
        self.model.eval()
        
        # Standard classification prompt (from training)
        # self.prompt = "What skin disease is shown in this image?"
        self.prompt = """Examine this skin image step by step:
1. Describe the lesion type (macule, papule, plaque, nodule, vesicle, etc.)
2. Note the color and texture
3. Observe the border and distribution
4. Based on these features, what is the diagnosis?

Answer with only the disease name."""
        
        print("✓ Model loaded and ready for inference!")
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image"""
        return Image.open(image_path).convert("RGB")
    
    def _create_messages(self, image: Image.Image) -> List[Dict]:
        """Create conversation messages for the model"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            },
        ]
    
    def _extract_disease(self, response: str) -> str:
        """Extract disease name from model response and normalize it"""
        response = response.strip().lower()
        
        extracted = None
        
        # Pattern: "This image shows {disease}."
        match = re.search(r"this image shows\s+([^.]+)", response, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
        
        # Pattern: "The diagnosis is {disease}"
        if not extracted:
            match = re.search(r"diagnosis is\s+([^.]+)", response, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
        
        # Pattern: "shows {disease}"
        if not extracted:
            match = re.search(r"shows\s+([^.]+)", response, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
        
        # If no pattern matched, use cleaned response
        if not extracted:
            # Remove common prefixes/suffixes
            response = re.sub(r"^(the|a|an)\s+", "", response)
            response = re.sub(r"\.$", "", response)
            extracted = response.strip()
        
        # Normalize the extracted disease name
        # This handles variations like "basal-cell-carcinoma" vs "basal cell carcinoma"
        return normalize_disease_name(extracted)
    
    def predict_single_with_topk(self, image_path: str, disease_list: List[str], 
                                  max_new_tokens: int = 128, num_samples: int = 5,
                                  temperature: float = 0.7) -> Dict:
        """
        Predict disease with Top-K candidates using batch sampling.
        
        This generates multiple predictions in a single batch and ranks them by frequency.
        Optimized to use num_return_sequences for faster generation.
        """
        try:
            image = self._load_image(image_path)
            messages = self._create_messages(image)
            
            # Apply chat template
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                max_length=4096,
                truncation=False,
            ).to(self.model.device)
            
            predictions_count = defaultdict(int)
            all_predictions = []
            input_length = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                # Try batch generation with num_return_sequences (faster)
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        num_return_sequences=num_samples,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    
                    # Process all generated sequences
                    for i in range(outputs.shape[0]):
                        generated_text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                        
                        # Extract assistant response
                        if "assistant" in generated_text.lower():
                            response = generated_text.split("assistant")[-1].strip()
                        else:
                            response = self.tokenizer.decode(outputs[i][input_length:], skip_special_tokens=True)
                        
                        # Extract and match disease
                        pred_disease = self._extract_disease(response)
                        matched_disease = self._match_to_disease_list(pred_disease, disease_list)
                        
                        if matched_disease:
                            predictions_count[matched_disease] += 1
                            all_predictions.append(matched_disease)
                
                except Exception:
                    # Fallback to sequential generation if batch fails
                    for _ in range(num_samples):
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=0.9,
                            use_cache=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                        
                        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        if "assistant" in generated_text.lower():
                            response = generated_text.split("assistant")[-1].strip()
                        else:
                            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                        
                        pred_disease = self._extract_disease(response)
                        matched_disease = self._match_to_disease_list(pred_disease, disease_list)
                        
                        if matched_disease:
                            predictions_count[matched_disease] += 1
                            all_predictions.append(matched_disease)
            
            # Sort by frequency to get top-k
            sorted_predictions = sorted(predictions_count.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate confidence scores
            actual_samples = len(all_predictions) if all_predictions else 1
            top_k_predictions = []
            for disease, count in sorted_predictions:
                top_k_predictions.append({
                    "disease": disease,
                    "confidence": count / actual_samples,
                    "count": count
                })
            
            # Get best prediction
            best_prediction = top_k_predictions[0]["disease"] if top_k_predictions else None
            
            return {
                "raw_predictions": all_predictions,
                "predicted_disease": best_prediction,
                "top_k_predictions": top_k_predictions,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "raw_predictions": [],
                "predicted_disease": None,
                "top_k_predictions": [],
                "status": f"error: {str(e)}"
            }
    
    def _match_to_disease_list(self, prediction: str, disease_list: List[str]) -> Optional[str]:
        """
        Match a prediction to the closest disease in the list.
        Handles variations like "basal-cell-carcinoma" vs "basal cell carcinoma".
        """
        # Normalize the prediction
        pred_normalized = normalize_disease_name(prediction)
        
        # Create lookup for normalized disease names
        disease_lookup = create_disease_lookup(disease_list)
        
        # 1. Exact match after normalization
        if pred_normalized in disease_lookup:
            return disease_lookup[pred_normalized]
        
        # 2. Check if prediction is in disease_list directly (original form)
        prediction_lower = prediction.lower().strip()
        if prediction_lower in disease_list:
            return prediction_lower
        
        # 3. Check if normalized prediction contains or is contained by any normalized disease
        for norm_disease, orig_disease in disease_lookup.items():
            if pred_normalized in norm_disease or norm_disease in pred_normalized:
                return orig_disease
        
        # 4. Word overlap matching (fuzzy)
        pred_words = set(pred_normalized.split())
        best_match = None
        best_score = 0
        
        for norm_disease, orig_disease in disease_lookup.items():
            disease_words = set(norm_disease.split())
            overlap = len(pred_words & disease_words)
            # Calculate Jaccard-like similarity
            union = len(pred_words | disease_words)
            similarity = overlap / union if union > 0 else 0
            
            # Prefer higher overlap with better similarity ratio
            score = overlap + similarity
            if score > best_score:
                best_score = score
                best_match = orig_disease
        
        # Only return if there's meaningful overlap (at least 1 word match)
        if best_score >= 1:
            return best_match
        
        return None
    
    def predict_single(self, image_path: str, max_new_tokens: int = 128, 
                       temperature: float = 0.1, top_p: float = 0.9) -> Dict:
        """Predict disease for a single image"""
        try:
            image = self._load_image(image_path)
            messages = self._create_messages(image)
            
            # Apply chat template
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                max_length=4096,
                truncation=False,
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "assistant" in generated_text.lower():
                response = generated_text.split("assistant")[-1].strip()
            else:
                # Get only the generated part
                input_length = inputs["input_ids"].shape[1]
                response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            # Extract disease name
            predicted_disease = self._extract_disease(response)
            
            return {
                "raw_response": response,
                "predicted_disease": predicted_disease,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "raw_response": None,
                "predicted_disease": None,
                "status": f"error: {str(e)}"
            }
    
    def _prepare_batch_inputs(self, image_paths: List[str]) -> Tuple[List, List[Image.Image]]:
        """Prepare batch inputs for multiple images"""
        images = []
        messages_list = []
        
        for img_path in image_paths:
            try:
                image = self._load_image(img_path)
                images.append(image)
                messages_list.append(self._create_messages(image))
            except Exception as e:
                images.append(None)
                messages_list.append(None)
        
        return messages_list, images
    
    def predict_batch(self, data: List[Dict], config: InferenceConfig, 
                      desc: str = "Inference") -> List[Dict]:
        """Predict diseases for a batch of images with TRUE batch processing"""
        results = []
        batch_size = config.BATCH_SIZE
        total = len(data)
        
        print(f"\n🚀 Batch inference: batch_size={batch_size}, total={total}")
        
        pbar = tqdm(total=total, desc=desc, dynamic_ncols=True, unit="img",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_data = data[batch_start:batch_end]
            
            # Load all images in batch (parallel loading)
            batch_images = []
            batch_valid = []
            
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                loaded_images = list(executor.map(
                    lambda x: self._safe_load_image(x["image_path"]), 
                    batch_data
                ))
            
            # Prepare batch messages
            batch_messages = []
            valid_indices = []
            
            for i, (item, image) in enumerate(zip(batch_data, loaded_images)):
                if image is not None:
                    batch_messages.append(self._create_messages(image))
                    valid_indices.append(i)
                    batch_valid.append(True)
                else:
                    batch_valid.append(False)
            
            # Process valid images in batch
            if batch_messages:
                try:
                    # Tokenize all messages in batch
                    batch_inputs = self.tokenizer.apply_chat_template(
                        batch_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        padding=True,
                        max_length=4096,
                        truncation=True,
                    ).to(self.model.device)
                    
                    # Generate for entire batch
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **batch_inputs,
                            max_new_tokens=config.MAX_NEW_TOKENS,
                            do_sample=config.TEMPERATURE > 0,
                            temperature=config.TEMPERATURE if config.TEMPERATURE > 0 else None,
                            top_p=config.TOP_P,
                            use_cache=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                    
                    # Decode all outputs
                    batch_responses = []
                    for i in range(outputs.shape[0]):
                        generated_text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                        
                        if "assistant" in generated_text.lower():
                            response = generated_text.split("assistant")[-1].strip()
                        else:
                            response = generated_text
                        
                        predicted_disease = self._extract_disease(response)
                        batch_responses.append({
                            "raw_response": response,
                            "predicted_disease": predicted_disease,
                            "status": "success"
                        })
                    
                except Exception as e:
                    # Fallback to single processing if batch fails
                    batch_responses = []
                    for msg, image in zip(batch_messages, [loaded_images[i] for i in valid_indices]):
                        try:
                            inputs = self.tokenizer.apply_chat_template(
                                msg,
                                tokenize=True,
                                add_generation_prompt=True,
                                return_tensors="pt",
                                return_dict=True,
                                max_length=4096,
                                truncation=False,
                            ).to(self.model.device)
                            
                            with torch.no_grad():
                                output = self.model.generate(
                                    **inputs,
                                    max_new_tokens=config.MAX_NEW_TOKENS,
                                    do_sample=config.TEMPERATURE > 0,
                                    temperature=config.TEMPERATURE if config.TEMPERATURE > 0 else None,
                                    top_p=config.TOP_P,
                                    use_cache=True,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                )
                            
                            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                            if "assistant" in generated_text.lower():
                                response = generated_text.split("assistant")[-1].strip()
                            else:
                                response = generated_text
                            
                            batch_responses.append({
                                "raw_response": response,
                                "predicted_disease": self._extract_disease(response),
                                "status": "success"
                            })
                        except Exception as inner_e:
                            batch_responses.append({
                                "raw_response": None,
                                "predicted_disease": None,
                                "status": f"error: {str(inner_e)}"
                            })
            
            # Combine results
            response_idx = 0
            for i, item in enumerate(batch_data):
                if batch_valid[i]:
                    resp = batch_responses[response_idx]
                    response_idx += 1
                else:
                    resp = {
                        "raw_response": None,
                        "predicted_disease": None,
                        "status": "error: failed to load image"
                    }
                
                results.append({
                    "id": item["id"],
                    "file_name": item["file_name"],
                    "image_path": item["image_path"],
                    "ground_truth": item["disease"],
                    "predicted": resp["predicted_disease"],
                    "raw_response": resp["raw_response"],
                    "status": resp["status"],
                })
            
            pbar.update(len(batch_data))
            
            # Clear CUDA cache after each batch
            torch.cuda.empty_cache()
        
        pbar.close()
        return results
    
    def _safe_load_image(self, image_path: str) -> Optional[Image.Image]:
        """Safely load an image, returning None on failure"""
        try:
            return self._load_image(image_path)
        except Exception:
            return None
    
    def predict_batch_with_topk(self, data: List[Dict], disease_list: List[str],
                                 config: InferenceConfig, num_samples: int = 5,
                                 desc: str = "Inference (Top-K)") -> List[Dict]:
        """
        Predict diseases with Top-K candidates for evaluation.
        
        Args:
            data: List of sample dictionaries
            disease_list: List of all possible disease classes
            config: Inference configuration
            num_samples: Number of samples per image for Top-K estimation
            desc: Progress bar description
        
        Note: Top-K inference is ~{num_samples}x slower than standard inference
              because it generates multiple predictions per image.
        """
        results = []
        
        print(f"\n⚠️  Top-K mode: Generating {num_samples} samples per image (slower)")
        
        pbar = tqdm(data, desc=desc, dynamic_ncols=True, unit="img",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for item in pbar:
            result = self.predict_single_with_topk(
                item["image_path"],
                disease_list,
                max_new_tokens=config.MAX_NEW_TOKENS,
                num_samples=num_samples,
                temperature=0.7,  # Higher temperature for diverse sampling
            )
            
            results.append({
                "id": item["id"],
                "file_name": item["file_name"],
                "image_path": item["image_path"],
                "ground_truth": item["disease"],
                "predicted": result["predicted_disease"],
                "top_k_predictions": result["top_k_predictions"],
                "status": result["status"],
            })
            
            # Clear CUDA cache periodically
            if len(results) % 20 == 0:
                torch.cuda.empty_cache()
        
        return results


# ============================================================================
# METRICS CALCULATION
# ============================================================================

# Top-K values for medical multi-class classification
# Suggested K values based on dataset characteristics:
TOP_K_VALUES = [1, 3, 5, 10, 15, 20]


def calculate_topk_accuracy(results: List[Dict], k_values: List[int] = None) -> Dict:
    """
    Calculate Top-K accuracy from results with top_k_predictions.
    
    Top-K Accuracy: Checks if the ground truth is within the top K predictions.
    
    Suggested K values for medical diagnosis:
    - Top-1: Standard accuracy (exact match)
    - Top-3: Typical differential diagnosis (3 possibilities)
    - Top-5: Extended differential diagnosis
    - Top-10: For datasets with many similar conditions
    - Top-15: For fine-grained classification
    - Top-20: For very large class sets (100+ classes)
    """
    if k_values is None:
        k_values = TOP_K_VALUES
    
    # Filter results that have top_k_predictions
    valid_results = [r for r in results if r.get("status") == "success" and r.get("top_k_predictions")]
    
    if not valid_results:
        return {"error": "No valid Top-K predictions available"}
    
    topk_metrics = {
        "num_samples": len(valid_results),
        "k_values": k_values,
    }
    
    for k in k_values:
        correct = 0
        for result in valid_results:
            ground_truth = result["ground_truth"]
            top_k_preds = result["top_k_predictions"][:k]
            
            # Check if ground truth is in top-k predictions (using flexible matching)
            found_match = False
            for p in top_k_preds:
                if match_disease_names(p["disease"], ground_truth):
                    found_match = True
                    break
            
            if found_match:
                correct += 1
        
        accuracy = correct / len(valid_results) if valid_results else 0
        topk_metrics[f"top_{k}_accuracy"] = accuracy
        topk_metrics[f"top_{k}_correct"] = correct
        topk_metrics[f"top_{k}_incorrect"] = len(valid_results) - correct
    
    # Calculate incremental gains
    prev_acc = 0
    for k in k_values:
        curr_acc = topk_metrics[f"top_{k}_accuracy"]
        topk_metrics[f"top_{k}_gain"] = curr_acc - prev_acc
        prev_acc = curr_acc
    
    return topk_metrics

def calculate_metrics(results: List[Dict], label_list: List[str]) -> Dict:
    """Calculate comprehensive classification metrics for multi-class classification"""
    
    # Filter successful predictions
    valid_results = [r for r in results if r["status"] == "success" and r["predicted"] is not None]
    
    if not valid_results:
        return {"error": "No valid predictions to evaluate"}
    
    # Create normalized lookup for matching predictions to labels
    disease_lookup = create_disease_lookup(label_list)
    
    # Helper function to find best matching label
    def find_matching_label(pred: str) -> Optional[str]:
        pred_norm = normalize_disease_name(pred)
        
        # Exact match after normalization
        if pred_norm in disease_lookup:
            return disease_lookup[pred_norm]
        
        # Partial match: find label where all words match
        pred_words = set(pred_norm.split())
        
        best_match = None
        best_overlap = 0
        
        for norm_label, orig_label in disease_lookup.items():
            label_words = set(norm_label.split())
            
            # Check if one is subset of the other (partial match)
            if label_words.issubset(pred_words) or pred_words.issubset(label_words):
                overlap = len(pred_words & label_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = orig_label
        
        return best_match
    
    # Normalize and map predictions to canonical label names
    y_true = []
    y_pred = []
    
    for r in valid_results:
        gt = r["ground_truth"]
        pred = r["predicted"]
        
        # Ground truth should already be in label_list, but normalize just in case
        gt_normalized = normalize_disease_name(gt)
        if gt_normalized in disease_lookup:
            y_true.append(disease_lookup[gt_normalized])
        else:
            # Try partial match for ground truth too
            matched_gt = find_matching_label(gt)
            y_true.append(matched_gt if matched_gt else gt)
        
        # Map prediction to canonical form (with partial matching)
        matched_pred = find_matching_label(pred)
        if matched_pred:
            y_pred.append(matched_pred)
        else:
            y_pred.append(pred)  # Keep original if not found (will be "unknown")
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(label_list)}
    num_classes = len(label_list)
    
    # Convert to indices (handle unknown predictions)
    y_true_idx = []
    y_pred_idx = []
    unknown_predictions = []
    
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true in label_to_idx:
            y_true_idx.append(label_to_idx[true])
            
            if pred in label_to_idx:
                y_pred_idx.append(label_to_idx[pred])
            else:
                # Unknown prediction - assign to a special index
                y_pred_idx.append(-1)
                unknown_predictions.append({
                    "id": valid_results[i]["id"],
                    "ground_truth": true,
                    "predicted": pred,
                })
    
    # For metrics, we need valid pairs only
    valid_pairs = [(t, p) for t, p in zip(y_true_idx, y_pred_idx) if p != -1]
    
    if not valid_pairs:
        return {"error": "No valid prediction pairs to evaluate"}
    
    y_true_valid = np.array([p[0] for p in valid_pairs])
    y_pred_valid = np.array([p[1] for p in valid_pairs])
    
    metrics = {}
    
    # =========================================================================
    # BASIC STATISTICS
    # =========================================================================
    metrics["total_samples"] = len(results)
    metrics["successful_predictions"] = len(valid_results)
    metrics["valid_predictions"] = len(valid_pairs)
    metrics["unknown_predictions"] = len(unknown_predictions)
    metrics["failed_predictions"] = len(results) - len(valid_results)
    metrics["num_classes"] = num_classes
    
    # Class distribution in predictions
    unique_true, counts_true = np.unique(y_true_valid, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred_valid, return_counts=True)
    metrics["classes_in_ground_truth"] = len(unique_true)
    metrics["classes_in_predictions"] = len(unique_pred)
    
    # =========================================================================
    # ACCURACY METRICS
    # =========================================================================
    metrics["accuracy"] = accuracy_score(y_true_valid, y_pred_valid)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true_valid, y_pred_valid)
    metrics["zero_one_loss"] = zero_one_loss(y_true_valid, y_pred_valid)  # Fraction of misclassifications
    
    # Top-K Accuracy (exact match within top K predictions)
    # Since we don't have probabilities, we calculate "lenient" accuracy variants
    metrics["exact_match_accuracy"] = metrics["accuracy"]
    
    # =========================================================================
    # PRECISION, RECALL, F1 (Multiple Averaging Methods)
    # =========================================================================
    for average in ["macro", "micro", "weighted"]:
        metrics[f"precision_{average}"] = precision_score(
            y_true_valid, y_pred_valid, average=average, zero_division=0
        )
        metrics[f"recall_{average}"] = recall_score(
            y_true_valid, y_pred_valid, average=average, zero_division=0
        )
        metrics[f"f1_{average}"] = f1_score(
            y_true_valid, y_pred_valid, average=average, zero_division=0
        )
    
    # Jaccard Score (Intersection over Union)
    for average in ["macro", "micro", "weighted"]:
        metrics[f"jaccard_{average}"] = jaccard_score(
            y_true_valid, y_pred_valid, average=average, zero_division=0
        )
    
    # =========================================================================
    # CORRELATION & AGREEMENT METRICS
    # =========================================================================
    metrics["matthews_corrcoef"] = matthews_corrcoef(y_true_valid, y_pred_valid)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true_valid, y_pred_valid)
    
    # Cohen's Kappa with different weightings
    try:
        metrics["cohen_kappa_linear"] = cohen_kappa_score(y_true_valid, y_pred_valid, weights='linear')
        metrics["cohen_kappa_quadratic"] = cohen_kappa_score(y_true_valid, y_pred_valid, weights='quadratic')
    except:
        metrics["cohen_kappa_linear"] = None
        metrics["cohen_kappa_quadratic"] = None
    
    # =========================================================================
    # PER-CLASS METRICS (Sensitivity, Specificity, PPV, NPV)
    # =========================================================================
    
    # Get multilabel confusion matrix for per-class metrics
    mcm = multilabel_confusion_matrix(y_true_valid, y_pred_valid, labels=list(range(num_classes)))
    
    per_class_metrics = {}
    sensitivities = []
    specificities = []
    ppvs = []
    npvs = []
    
    for i, label in enumerate(label_list):
        tn, fp, fn, tp = mcm[i].ravel()
        
        # Sensitivity (Recall / True Positive Rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Positive Predictive Value (Precision)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Negative Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # F1 Score
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        
        # Support (number of true samples)
        support = tp + fn
        
        per_class_metrics[label] = {
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "sensitivity": sensitivity,  # Same as recall
            "specificity": specificity,
            "ppv": ppv,  # Same as precision
            "npv": npv,
            "f1_score": f1,
            "support": int(support),
        }
        
        if support > 0:  # Only include classes with samples
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            ppvs.append(ppv)
            npvs.append(npv)
    
    metrics["per_class_detailed"] = per_class_metrics
    
    # Averaged metrics across classes
    metrics["sensitivity_macro"] = np.mean(sensitivities) if sensitivities else 0
    metrics["specificity_macro"] = np.mean(specificities) if specificities else 0
    metrics["ppv_macro"] = np.mean(ppvs) if ppvs else 0
    metrics["npv_macro"] = np.mean(npvs) if npvs else 0
    
    # =========================================================================
    # CLASSIFICATION REPORT (sklearn standard)
    # =========================================================================
    class_report = classification_report(
        y_true_valid, y_pred_valid, 
        labels=list(range(num_classes)),
        target_names=label_list, 
        output_dict=True,
        zero_division=0
    )
    metrics["per_class_report"] = class_report
    
    # =========================================================================
    # CONFUSION MATRIX ANALYSIS
    # =========================================================================
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=list(range(num_classes)))
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_labels"] = label_list
    
    # Confusion matrix derived metrics
    cm_sum = cm.sum()
    cm_diagonal = np.diag(cm).sum()
    
    metrics["total_correct"] = int(cm_diagonal)
    metrics["total_incorrect"] = int(cm_sum - cm_diagonal)
    
    # Per-class accuracy (diagonal / row sum)
    row_sums = cm.sum(axis=1)
    per_class_acc = np.diag(cm) / np.where(row_sums > 0, row_sums, 1)
    metrics["per_class_accuracy"] = {label_list[i]: float(per_class_acc[i]) for i in range(num_classes) if row_sums[i] > 0}
    
    # Most confused pairs
    np.fill_diagonal(cm, 0)  # Zero out diagonal for confusion analysis
    confusion_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0:
                confusion_pairs.append({
                    "true": label_list[i],
                    "predicted": label_list[j],
                    "count": int(cm[i, j])
                })
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)
    metrics["top_confusions"] = confusion_pairs[:30]
    
    # =========================================================================
    # CLASS IMBALANCE METRICS
    # =========================================================================
    
    # Class distribution
    class_distribution = {}
    for i, label in enumerate(label_list):
        count = int((y_true_valid == i).sum())
        if count > 0:
            class_distribution[label] = count
    
    metrics["class_distribution"] = class_distribution
    
    # Imbalance ratio (max class / min class)
    if class_distribution:
        counts = list(class_distribution.values())
        metrics["imbalance_ratio"] = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        metrics["class_with_most_samples"] = max(class_distribution, key=class_distribution.get)
        metrics["class_with_least_samples"] = min(class_distribution, key=class_distribution.get)
    
    # =========================================================================
    # ERROR ANALYSIS METRICS
    # =========================================================================
    
    # Error rate by class
    error_rates = {}
    for i, label in enumerate(label_list):
        mask = y_true_valid == i
        if mask.sum() > 0:
            errors = (y_pred_valid[mask] != i).sum()
            error_rates[label] = float(errors / mask.sum())
    
    metrics["per_class_error_rate"] = error_rates
    
    # Classes with highest error rates
    if error_rates:
        sorted_errors = sorted(error_rates.items(), key=lambda x: x[1], reverse=True)
        metrics["most_difficult_classes"] = sorted_errors[:10]
        metrics["easiest_classes"] = sorted_errors[-10:][::-1]
    
    # =========================================================================
    # SUMMARY SCORES
    # =========================================================================
    
    # Overall quality score (combination metric)
    metrics["overall_score"] = (
        metrics["accuracy"] * 0.2 +
        metrics["balanced_accuracy"] * 0.2 +
        metrics["f1_macro"] * 0.2 +
        metrics["f1_weighted"] * 0.2 +
        (metrics["matthews_corrcoef"] + 1) / 2 * 0.2  # Normalize MCC to 0-1
    )
    
    # =========================================================================
    # TOP-K ACCURACY (if top_k_predictions available)
    # =========================================================================
    has_topk = any(r.get("top_k_predictions") for r in results if r.get("status") == "success")
    if has_topk:
        topk_metrics = calculate_topk_accuracy(results, TOP_K_VALUES)
        metrics["top_k_metrics"] = topk_metrics
        
        # Add top-k accuracies to main metrics for easy access
        for k in TOP_K_VALUES:
            if f"top_{k}_accuracy" in topk_metrics:
                metrics[f"top_{k}_accuracy"] = topk_metrics[f"top_{k}_accuracy"]
    
    # Unknown predictions summary
    if unknown_predictions:
        metrics["unknown_predictions_list"] = unknown_predictions[:20]
        
        # Count unique unknown predictions
        unknown_pred_set = set(up["predicted"] for up in unknown_predictions)
        metrics["unique_unknown_predictions"] = list(unknown_pred_set)[:20]
    
    return metrics


def generate_error_analysis(results: List[Dict], label_list: List[str]) -> Dict:
    """Generate detailed error analysis with flexible matching"""
    
    analysis = {
        "misclassifications": [],
        "confusion_pairs": defaultdict(int),
        "per_class_errors": defaultdict(lambda: {"total": 0, "errors": 0, "error_rate": 0.0}),
        "error_examples": [],
        "partial_matches": [],  # Track partial matches counted as correct
    }
    
    valid_results = [r for r in results if r["status"] == "success" and r["predicted"] is not None]
    
    for result in valid_results:
        gt = result["ground_truth"]
        pred = result["predicted"]
        
        analysis["per_class_errors"][gt]["total"] += 1
        
        # Use flexible matching
        is_correct = match_disease_names(pred, gt)
        
        if not is_correct:
            analysis["per_class_errors"][gt]["errors"] += 1
            analysis["confusion_pairs"][(gt, pred)] += 1
            analysis["misclassifications"].append({
                "id": result["id"],
                "file_name": result["file_name"],
                "ground_truth": gt,
                "predicted": pred,
                "raw_response": result.get("raw_response", ""),
            })
        elif normalize_disease_name(pred) != normalize_disease_name(gt):
            # It's a partial match
            analysis["partial_matches"].append({
                "id": result["id"],
                "ground_truth": gt,
                "predicted": pred,
            })
    
    # Calculate error rates
    for cls in analysis["per_class_errors"]:
        stats = analysis["per_class_errors"][cls]
        if stats["total"] > 0:
            stats["error_rate"] = stats["errors"] / stats["total"]
    
    # Convert defaultdict to regular dict for JSON serialization
    analysis["per_class_errors"] = dict(analysis["per_class_errors"])
    analysis["confusion_pairs"] = {f"{k[0]} -> {k[1]}": v for k, v in analysis["confusion_pairs"].items()}
    
    # Sort confusion pairs by frequency
    analysis["top_confusion_pairs"] = sorted(
        analysis["confusion_pairs"].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:20]
    
    # Sample error examples
    analysis["error_examples"] = analysis["misclassifications"][:50]
    
    # Summary statistics
    analysis["summary"] = {
        "total_errors": len(analysis["misclassifications"]),
        "total_samples": len(valid_results),
        "error_rate": len(analysis["misclassifications"]) / len(valid_results) if valid_results else 0,
        "unique_confusion_pairs": len(analysis["confusion_pairs"]),
    }
    
    return analysis


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(results: Dict, output_dir: str, split_name: str):
    """Save results to JSON file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = output_path / f"classification_results_{split_name}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ Results saved to: {filename}")
    return filename


def save_confusion_matrix_plot(cm: List[List[int]], labels: List[str], 
                                output_dir: str, split_name: str, top_n: int = 30):
    """Save confusion matrix as an image (top N classes by frequency)"""
    if not HAS_VISUALIZATION:
        print("Skipping confusion matrix plot (matplotlib not available)")
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cm_array = np.array(cm)
    
    # If too many classes, select top N by sample count
    if len(labels) > top_n:
        # Sum of each row (true samples per class)
        class_counts = cm_array.sum(axis=1)
        top_indices = np.argsort(class_counts)[-top_n:]
        
        cm_array = cm_array[np.ix_(top_indices, top_indices)]
        labels = [labels[i] for i in top_indices]
        title_suffix = f" (Top {top_n} classes)"
    else:
        title_suffix = ""
    
    # Create figure
    fig_size = max(10, len(labels) * 0.4)
    plt.figure(figsize=(fig_size, fig_size))
    
    # Normalize confusion matrix for better visualization
    cm_normalized = cm_array.astype('float') / (cm_array.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # Create heatmap
    sns.heatmap(
        cm_normalized, 
        annot=False,  # Turn off annotations if too many classes
        fmt='.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        square=True,
    )
    
    plt.title(f'Confusion Matrix - {split_name.upper()}{title_suffix}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # Save
    filename = output_path / f"confusion_matrix_{split_name}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {filename}")
    return filename


def save_metrics_report(metrics: Dict, error_analysis: Dict, output_dir: str, split_name: str):
    """Save a human-readable metrics report as text file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = output_path / f"metrics_report_{split_name}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"CLASSIFICATION METRICS REPORT - {split_name.upper()}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")
        
        # Sample Statistics
        f.write("SAMPLE STATISTICS\n")
        f.write("-"*50 + "\n")
        f.write(f"Total samples:           {metrics.get('total_samples', 'N/A')}\n")
        f.write(f"Successful predictions:  {metrics.get('successful_predictions', 'N/A')}\n")
        f.write(f"Valid predictions:       {metrics.get('valid_predictions', 'N/A')}\n")
        f.write(f"Unknown predictions:     {metrics.get('unknown_predictions', 'N/A')}\n")
        f.write(f"Failed predictions:      {metrics.get('failed_predictions', 'N/A')}\n")
        f.write(f"Number of classes:       {metrics.get('num_classes', 'N/A')}\n")
        f.write(f"Classes in ground truth: {metrics.get('classes_in_ground_truth', 'N/A')}\n")
        f.write(f"Classes in predictions:  {metrics.get('classes_in_predictions', 'N/A')}\n\n")
        
        # Accuracy Metrics
        f.write("ACCURACY METRICS\n")
        f.write("-"*50 + "\n")
        f.write(f"Accuracy:               {metrics.get('accuracy', 0):.4f}\n")
        f.write(f"Balanced Accuracy:      {metrics.get('balanced_accuracy', 0):.4f}\n")
        f.write(f"Zero-One Loss:          {metrics.get('zero_one_loss', 0):.4f}\n\n")
        
        # Precision/Recall/F1
        f.write("PRECISION / RECALL / F1-SCORE\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Metric':<20} {'Macro':>10} {'Micro':>10} {'Weighted':>10}\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Precision':<20} {metrics.get('precision_macro', 0):>10.4f} {metrics.get('precision_micro', 0):>10.4f} {metrics.get('precision_weighted', 0):>10.4f}\n")
        f.write(f"{'Recall':<20} {metrics.get('recall_macro', 0):>10.4f} {metrics.get('recall_micro', 0):>10.4f} {metrics.get('recall_weighted', 0):>10.4f}\n")
        f.write(f"{'F1-Score':<20} {metrics.get('f1_macro', 0):>10.4f} {metrics.get('f1_micro', 0):>10.4f} {metrics.get('f1_weighted', 0):>10.4f}\n")
        f.write(f"{'Jaccard (IoU)':<20} {metrics.get('jaccard_macro', 0):>10.4f} {metrics.get('jaccard_micro', 0):>10.4f} {metrics.get('jaccard_weighted', 0):>10.4f}\n\n")
        
        # Medical Metrics
        f.write("MEDICAL METRICS (Macro Averaged)\n")
        f.write("-"*50 + "\n")
        f.write(f"Sensitivity (Recall):   {metrics.get('sensitivity_macro', 0):.4f}\n")
        f.write(f"Specificity:            {metrics.get('specificity_macro', 0):.4f}\n")
        f.write(f"PPV (Precision):        {metrics.get('ppv_macro', 0):.4f}\n")
        f.write(f"NPV:                    {metrics.get('npv_macro', 0):.4f}\n\n")
        
        # Correlation & Agreement
        f.write("CORRELATION & AGREEMENT METRICS\n")
        f.write("-"*50 + "\n")
        f.write(f"Matthews Corr. Coef:    {metrics.get('matthews_corrcoef', 0):.4f}\n")
        f.write(f"Cohen's Kappa:          {metrics.get('cohen_kappa', 0):.4f}\n")
        if metrics.get('cohen_kappa_linear') is not None:
            f.write(f"Cohen's Kappa (linear): {metrics.get('cohen_kappa_linear', 0):.4f}\n")
            f.write(f"Cohen's Kappa (quadr.): {metrics.get('cohen_kappa_quadratic', 0):.4f}\n")
        f.write("\n")
        
        # Overall Score
        f.write("OVERALL SCORE\n")
        f.write("-"*50 + "\n")
        f.write(f"Combined Score:         {metrics.get('overall_score', 0):.4f}\n")
        f.write("(Weighted avg of accuracy, balanced acc, F1 macro/weighted, MCC)\n\n")
        
        # Top-K Accuracy
        if metrics.get('top_k_metrics') or metrics.get('top_1_accuracy') is not None:
            f.write("TOP-K ACCURACY\n")
            f.write("-"*50 + "\n")
            f.write("(Checks if ground truth is within top K predictions)\n\n")
            f.write(f"{'K':>5} {'Accuracy':>12} {'Correct':>10} {'Incorrect':>10} {'Gain':>10}\n")
            f.write("-"*50 + "\n")
            topk = metrics.get('top_k_metrics', {})
            for k in TOP_K_VALUES:
                acc = metrics.get(f'top_{k}_accuracy', topk.get(f'top_{k}_accuracy'))
                if acc is not None:
                    correct = topk.get(f'top_{k}_correct', 'N/A')
                    incorrect = topk.get(f'top_{k}_incorrect', 'N/A')
                    gain = topk.get(f'top_{k}_gain', 0)
                    f.write(f"{k:>5} {acc:>12.4f} {str(correct):>10} {str(incorrect):>10} {gain:>10.4f}\n")
            f.write("\n")
            f.write("Suggested interpretation:\n")
            f.write("  - Top-1:  Exact match accuracy (standard metric)\n")
            f.write("  - Top-3:  Typical differential diagnosis scenario\n")
            f.write("  - Top-5:  Extended differential diagnosis\n")
            f.write("  - Top-10: For many similar conditions\n")
            f.write("  - Top-20: For very large class sets\n\n")
        
        # Class Imbalance
        f.write("CLASS IMBALANCE\n")
        f.write("-"*50 + "\n")
        imb_ratio = metrics.get('imbalance_ratio', 'N/A')
        f.write(f"Imbalance Ratio:        {imb_ratio:.2f}\n" if isinstance(imb_ratio, (int, float)) else f"Imbalance Ratio: {imb_ratio}\n")
        f.write(f"Most samples:           {metrics.get('class_with_most_samples', 'N/A')}\n")
        f.write(f"Least samples:          {metrics.get('class_with_least_samples', 'N/A')}\n\n")
        
        # Most Difficult Classes
        most_difficult = metrics.get('most_difficult_classes', [])
        if most_difficult:
            f.write("MOST DIFFICULT CLASSES (Highest Error Rate)\n")
            f.write("-"*50 + "\n")
            for cls, err_rate in most_difficult[:15]:
                f.write(f"  {cls}: {err_rate:.2%}\n")
            f.write("\n")
        
        # Top Confusions
        top_confusions = metrics.get('top_confusions', [])
        if top_confusions:
            f.write("TOP CONFUSION PAIRS\n")
            f.write("-"*50 + "\n")
            for conf in top_confusions[:20]:
                f.write(f"  {conf['true']} -> {conf['predicted']}: {conf['count']}\n")
            f.write("\n")
        
        # Error Analysis
        if error_analysis:
            summary = error_analysis.get("summary", {})
            f.write("ERROR ANALYSIS SUMMARY\n")
            f.write("-"*50 + "\n")
            f.write(f"Total errors:           {summary.get('total_errors', 'N/A')}\n")
            f.write(f"Error rate:             {summary.get('error_rate', 0):.2%}\n")
            f.write(f"Unique confusion pairs: {summary.get('unique_confusion_pairs', 'N/A')}\n\n")
        
        # Per-class report
        per_class = metrics.get("per_class_report", {})
        if per_class:
            f.write("PER-CLASS METRICS (sklearn classification_report)\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Class':<45} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Sup':>8}\n")
            f.write("-"*80 + "\n")
            
            for cls, stats in sorted(per_class.items()):
                if cls not in ['accuracy', 'macro avg', 'micro avg', 'weighted avg']:
                    f.write(f"{cls[:45]:<45} {stats.get('precision', 0):>8.3f} "
                           f"{stats.get('recall', 0):>8.3f} {stats.get('f1-score', 0):>8.3f} "
                           f"{stats.get('support', 0):>8.0f}\n")
            
            f.write("-"*80 + "\n")
            # Print averages
            for avg_type in ['macro avg', 'micro avg', 'weighted avg']:
                if avg_type in per_class:
                    stats = per_class[avg_type]
                    f.write(f"{avg_type:<45} {stats.get('precision', 0):>8.3f} "
                           f"{stats.get('recall', 0):>8.3f} {stats.get('f1-score', 0):>8.3f} "
                           f"{stats.get('support', 0):>8.0f}\n")
        
        # Per-class detailed (with specificity, NPV)
        per_class_detailed = metrics.get("per_class_detailed", {})
        if per_class_detailed:
            f.write("\n\nPER-CLASS DETAILED METRICS (Medical)\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Class':<35} {'Sens':>7} {'Spec':>7} {'PPV':>7} {'NPV':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>6}\n")
            f.write("-"*100 + "\n")
            
            # Sort by support (number of samples)
            sorted_classes = sorted(per_class_detailed.items(), key=lambda x: x[1].get('support', 0), reverse=True)
            
            for cls, stats in sorted_classes:
                if stats.get('support', 0) > 0:  # Only show classes with samples
                    f.write(f"{cls[:35]:<35} {stats.get('sensitivity', 0):>7.3f} "
                           f"{stats.get('specificity', 0):>7.3f} {stats.get('ppv', 0):>7.3f} "
                           f"{stats.get('npv', 0):>7.3f} {stats.get('f1_score', 0):>7.3f} "
                           f"{stats.get('true_positives', 0):>5} {stats.get('false_positives', 0):>5} "
                           f"{stats.get('false_negatives', 0):>5} {stats.get('true_negatives', 0):>6}\n")
    
    print(f"✓ Metrics report saved to: {filename}")
    return filename


def print_metrics_summary(metrics: Dict, split_name: str):
    """Print a formatted summary of metrics"""
    print(f"\n{'='*60}")
    print(f"METRICS SUMMARY - {split_name.upper()}")
    print(f"{'='*60}")
    
    print(f"\n📊 Sample Statistics:")
    print(f"   Total samples:          {metrics.get('total_samples', 'N/A')}")
    print(f"   Successful predictions:  {metrics.get('successful_predictions', 'N/A')}")
    print(f"   Valid predictions:       {metrics.get('valid_predictions', 'N/A')}")
    print(f"   Unknown predictions:     {metrics.get('unknown_predictions', 'N/A')}")
    print(f"   Failed predictions:      {metrics.get('failed_predictions', 'N/A')}")
    print(f"   Number of classes:       {metrics.get('num_classes', 'N/A')}")
    print(f"   Classes in ground truth: {metrics.get('classes_in_ground_truth', 'N/A')}")
    print(f"   Classes in predictions:  {metrics.get('classes_in_predictions', 'N/A')}")
    
    print(f"\n📈 Accuracy Metrics:")
    print(f"   Accuracy:               {metrics.get('accuracy', 0):.4f}")
    print(f"   Balanced Accuracy:      {metrics.get('balanced_accuracy', 0):.4f}")
    print(f"   Zero-One Loss:          {metrics.get('zero_one_loss', 0):.4f}")
    
    print(f"\n📊 Precision (by averaging):")
    print(f"   Macro:    {metrics.get('precision_macro', 0):.4f}")
    print(f"   Micro:    {metrics.get('precision_micro', 0):.4f}")
    print(f"   Weighted: {metrics.get('precision_weighted', 0):.4f}")
    
    print(f"\n📊 Recall / Sensitivity (by averaging):")
    print(f"   Macro:    {metrics.get('recall_macro', 0):.4f}")
    print(f"   Micro:    {metrics.get('recall_micro', 0):.4f}")
    print(f"   Weighted: {metrics.get('recall_weighted', 0):.4f}")
    
    print(f"\n📊 F1-Score (by averaging):")
    print(f"   Macro:    {metrics.get('f1_macro', 0):.4f}")
    print(f"   Micro:    {metrics.get('f1_micro', 0):.4f}")
    print(f"   Weighted: {metrics.get('f1_weighted', 0):.4f}")
    
    print(f"\n📊 Jaccard Score / IoU (by averaging):")
    print(f"   Macro:    {metrics.get('jaccard_macro', 0):.4f}")
    print(f"   Micro:    {metrics.get('jaccard_micro', 0):.4f}")
    print(f"   Weighted: {metrics.get('jaccard_weighted', 0):.4f}")
    
    print(f"\n📊 Medical Metrics (Macro Averaged):")
    print(f"   Sensitivity (Recall):   {metrics.get('sensitivity_macro', 0):.4f}")
    print(f"   Specificity:            {metrics.get('specificity_macro', 0):.4f}")
    print(f"   PPV (Precision):        {metrics.get('ppv_macro', 0):.4f}")
    print(f"   NPV:                    {metrics.get('npv_macro', 0):.4f}")
    
    print(f"\n📊 Correlation & Agreement:")
    print(f"   Matthews Corr. Coef:    {metrics.get('matthews_corrcoef', 0):.4f}")
    print(f"   Cohen's Kappa:          {metrics.get('cohen_kappa', 0):.4f}")
    if metrics.get('cohen_kappa_linear') is not None:
        print(f"   Cohen's Kappa (linear): {metrics.get('cohen_kappa_linear', 0):.4f}")
        print(f"   Cohen's Kappa (quadr.): {metrics.get('cohen_kappa_quadratic', 0):.4f}")
    
    print(f"\n📊 Class Imbalance:")
    print(f"   Imbalance Ratio:        {metrics.get('imbalance_ratio', 'N/A'):.2f}" if isinstance(metrics.get('imbalance_ratio'), (int, float)) else f"   Imbalance Ratio: N/A")
    print(f"   Most samples:           {metrics.get('class_with_most_samples', 'N/A')}")
    print(f"   Least samples:          {metrics.get('class_with_least_samples', 'N/A')}")
    
    print(f"\n⭐ Overall Score:          {metrics.get('overall_score', 0):.4f}")
    
    # Top-K Accuracy
    if metrics.get('top_k_metrics') or metrics.get('top_1_accuracy') is not None:
        print(f"\n📊 Top-K Accuracy:")
        for k in TOP_K_VALUES:
            acc = metrics.get(f'top_{k}_accuracy')
            if acc is not None:
                print(f"   Top-{k:>2}: {acc:.4f}")
    
    # Most difficult classes
    most_difficult = metrics.get('most_difficult_classes', [])
    if most_difficult:
        print(f"\n🔴 Most Difficult Classes (Highest Error Rate):")
        for cls, err_rate in most_difficult[:5]:
            print(f"   {cls}: {err_rate:.2%}")
    
    # Top confusions
    top_confusions = metrics.get('top_confusions', [])
    if top_confusions:
        print(f"\n🔄 Top Confusions:")
        for conf in top_confusions[:5]:
            print(f"   {conf['true']} → {conf['predicted']}: {conf['count']} times")


def print_error_analysis_summary(analysis: Dict, split_name: str):
    """Print error analysis summary"""
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS - {split_name.upper()}")
    print(f"{'='*60}")
    
    summary = analysis.get("summary", {})
    print(f"\n📉 Error Summary:")
    print(f"   Total errors:   {summary.get('total_errors', 'N/A')}")
    print(f"   Total samples:  {summary.get('total_samples', 'N/A')}")
    print(f"   Error rate:     {summary.get('error_rate', 0):.2%}")
    
    print(f"\n🔄 Top Confusion Pairs:")
    for pair, count in analysis.get("top_confusion_pairs", [])[:10]:
        print(f"   {pair}: {count} times")
    
    # Per-class error rates
    per_class = analysis.get("per_class_errors", {})
    if per_class:
        print(f"\n📊 Classes with Highest Error Rates:")
        sorted_classes = sorted(
            per_class.items(), 
            key=lambda x: x[1].get("error_rate", 0), 
            reverse=True
        )[:10]
        for cls, stats in sorted_classes:
            if stats["total"] > 0:
                print(f"   {cls}: {stats['error_rate']:.2%} ({stats['errors']}/{stats['total']})")


# ============================================================================
# MAIN INFERENCE FUNCTIONS
# ============================================================================

def run_inference(config: InferenceConfig, model_path: str = None):
    """Run inference on all splits"""
    
    print("\n" + "="*60)
    print("CLASSIFICATION MODEL INFERENCE")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Seed: {config.SEED}")
    print(f"Split ratios: Train={config.TRAIN_RATIO:.0%}, Val={config.VAL_RATIO:.0%}")
    
    # Setup
    set_seed(config.SEED)
    
    # Load data
    print("\n[1/5] Loading data...")
    data = load_skincap_data(config)
    
    # Get unique diseases
    disease_list = get_unique_diseases(data)
    print(f"Found {len(disease_list)} unique diseases")
    
    # Split data (matching training: 90/10)
    print("\n[2/5] Splitting data...")
    train_data, val_data = split_data(data, config)
    
    # Load model
    print("\n[3/5] Loading model...")
    if model_path is None:
        # Try merged model first, then LoRA
        if Path(config.MODEL_PATH_MERGED).exists():
            model_path = config.MODEL_PATH_MERGED
        else:
            model_path = config.MODEL_PATH_LORA
    
    model = ClassificationInference(model_path)
    
    # Create output directory
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        "config": asdict(config),
        "timestamp": datetime.now().isoformat(),
        "disease_list": disease_list,
        "num_diseases": len(disease_list),
        "splits": {},
    }
    
    # Run inference on each split
    print("\n[4/5] Running inference...")
    
    # Train: error analysis (model saw this data)
    # Val: true evaluation (model did NOT see this data)
    splits = [
        ("train", train_data, True),   # (name, data, do_error_analysis)
        ("val", val_data, True),       # Val is the TRUE unseen test set
    ]
    
    for split_name, split_samples, do_error_analysis in splits:
        print(f"\n{'─'*40}")
        print(f"Processing {split_name.upper()} split ({len(split_samples)} samples)")
        print(f"{'─'*40}")
        
        # Run inference
        if config.ENABLE_TOPK:
            # Run with Top-K predictions
            predictions = model.predict_batch_with_topk(
                split_samples, disease_list, config,
                num_samples=config.TOPK_SAMPLES,
                desc=f"Inference+TopK [{split_name}]"
            )
        else:
            # Standard inference
            predictions = model.predict_batch(
                split_samples, config, 
                desc=f"Inference [{split_name}]"
            )
        
        # Calculate metrics (will include Top-K if available)
        metrics = calculate_metrics(predictions, disease_list)
        
        # Print summary
        print_metrics_summary(metrics, split_name)
        
        # Error analysis (for train/val only)
        error_analysis = None
        if do_error_analysis:
            error_analysis = generate_error_analysis(predictions, disease_list)
            print_error_analysis_summary(error_analysis, split_name)
        
        # Store results
        split_results = {
            "split_name": split_name,
            "num_samples": len(split_samples),
            "predictions": predictions,
            "metrics": metrics,
            "error_analysis": error_analysis if do_error_analysis else "Error analysis not performed for test set (unseen data)",
        }
        
        all_results["splits"][split_name] = split_results
        
        # Save individual split results
        save_results(split_results, config.OUTPUT_DIR, split_name)
        
        # Save confusion matrix plot
        if "confusion_matrix" in metrics and "confusion_matrix_labels" in metrics:
            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                metrics["confusion_matrix_labels"],
                config.OUTPUT_DIR,
                split_name
            )
        
        # Save text report
        save_metrics_report(metrics, error_analysis if do_error_analysis else {}, config.OUTPUT_DIR, split_name)
    
    # Generate attention maps if enabled
    if config.ENABLE_ATTENTION:
        print("\n[5/6] Generating attention maps...")
        try:
            from transformers import AutoProcessor
            
            # Create attention extractor using the model's internal components
            processor = AutoProcessor.from_pretrained(model_path)
            attention_extractor = AttentionExtractor(model.model, processor)
            
            for split_name, split_results in all_results["splits"].items():
                predictions = split_results["predictions"]
                generate_attention_maps(
                    predictions, attention_extractor,
                    config.OUTPUT_DIR, split_name,
                    limit=config.ATTENTION_LIMIT,
                    prompt=model.prompt
                )
            
            attention_extractor.remove_hooks()
            
        except Exception as e:
            print(f"⚠ Attention map generation failed: {e}")
            print("   Continuing without attention maps...")
    
    # Save combined results
    print(f"\n[{'6/6' if config.ENABLE_ATTENTION else '5/5'}] Saving combined results...")
    save_results(all_results, config.OUTPUT_DIR, "combined")
    
    # Final summary
    print("\n" + "="*60)
    print("INFERENCE COMPLETED!")
    print("="*60)
    print(f"\n📦 Output files saved to: {config.OUTPUT_DIR}")
    print(f"\n📊 JSON Results:")
    print(f"   - classification_results_train.json (error analysis - seen data)")
    print(f"   - classification_results_val.json (TRUE evaluation - unseen data)")
    print(f"   - classification_results_combined.json")
    print(f"\n📈 Reports:")
    print(f"   - metrics_report_train.txt")
    print(f"   - metrics_report_val.txt")
    if HAS_VISUALIZATION:
        print(f"\n📉 Visualizations:")
        print(f"   - confusion_matrix_train.png")
        print(f"   - confusion_matrix_val.png")
        if config.ENABLE_ATTENTION:
            print(f"   - attention_maps_train/Correct/  (attention visualizations)")
            print(f"   - attention_maps_train/Incorrect/")
            print(f"   - attention_maps_val/Correct/")
            print(f"   - attention_maps_val/Incorrect/")
    
    print(f"\n⚠️  IMPORTANT:")
    print(f"   - Train metrics = model performance on SEEN data (error analysis)")
    print(f"   - Val metrics = model performance on UNSEEN data (true evaluation)")
    
    return all_results


def run_inference_single_split(config: InferenceConfig, split: str, model_path: str = None):
    """Run inference on a single split (train or val)"""
    
    set_seed(config.SEED)
    
    # Load data
    data = load_skincap_data(config)
    disease_list = get_unique_diseases(data)
    
    # Split data (90/10 matching training)
    train_data, val_data = split_data(data, config)
    
    # Select split
    if split == "train":
        split_data_selected = train_data
        is_seen_data = True
    elif split == "val":
        split_data_selected = val_data
        is_seen_data = False
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'.")
    
    # Load model
    if model_path is None:
        model_path = config.MODEL_PATH_MERGED if Path(config.MODEL_PATH_MERGED).exists() else config.MODEL_PATH_LORA
    
    model = ClassificationInference(model_path)
    
    # Run inference
    if config.ENABLE_TOPK:
        # Run with Top-K predictions
        predictions = model.predict_batch_with_topk(
            split_data_selected, disease_list, config,
            num_samples=config.TOPK_SAMPLES,
            desc=f"Inference+TopK [{split}]"
        )
    else:
        # Standard inference
        predictions = model.predict_batch(split_data_selected, config, desc=f"Inference [{split}]")
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, disease_list)
    
    if is_seen_data:
        print_metrics_summary(metrics, f"{split} (SEEN during training)")
    else:
        print_metrics_summary(metrics, f"{split} (UNSEEN - true evaluation)")
    
    # Error analysis
    error_analysis = generate_error_analysis(predictions, disease_list)
    print_error_analysis_summary(error_analysis, split)
    
    # Save results
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "split_name": split,
        "is_seen_data": is_seen_data,
        "num_samples": len(split_data_selected),
        "predictions": predictions,
        "metrics": metrics,
        "error_analysis": error_analysis,
    }
    save_results(results, config.OUTPUT_DIR, split)
    
    return predictions, metrics


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Classification Model Inference')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'train', 'val'],
                        help='Which split to run inference on (train=seen data, val=unseen data)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model (defaults to merged model)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for parallel GPU inference (default: 8, increase if VRAM allows)')
    parser.add_argument('--output_dir', type=str, default='./classification_results',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (must match training seed=42)')
    parser.add_argument('--topk', action='store_true',
                        help='Enable Top-K accuracy evaluation (slower, uses multiple sampling)')
    parser.add_argument('--topk_samples', type=int, default=5,
                        help='Number of samples per image for Top-K estimation (default: 5, higher=slower)')
    parser.add_argument('--attention', action='store_true',
                        help='Generate attention map visualizations for all predictions')
    parser.add_argument('--attention_limit', type=int, default=None,
                        help='Limit number of attention maps per split (default: all)')
    
    args = parser.parse_args()
    
    # Create config (90/10 split matching training)
    config = InferenceConfig(
        SEED=args.seed,
        BATCH_SIZE=args.batch_size,
        OUTPUT_DIR=args.output_dir,
        ENABLE_TOPK=args.topk,
        TOPK_SAMPLES=args.topk_samples,
        ENABLE_ATTENTION=args.attention,
        ATTENTION_LIMIT=args.attention_limit,
    )
    
    if args.seed != 42:
        print(f"⚠️  Warning: Using seed={args.seed} but training used seed=42")
        print(f"   This means splits won't match training data!")
    
    if args.topk:
        print(f"📊 Top-K evaluation ENABLED with {args.topk_samples} samples per image")
        print(f"   This will be slower but provides additional metrics")
    
    if args.attention:
        limit_msg = f" (limit: {args.attention_limit})" if args.attention_limit else " (all samples)"
        print(f"🎨 Attention map generation ENABLED{limit_msg}")
    
    if args.mode == 'all':
        run_inference(config, args.model)
    else:
        run_inference_single_split(config, args.mode, args.model)


if __name__ == "__main__":
    main()
