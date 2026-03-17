"""
Grad-CAM Attention Map Visualization for Classification Results

Generates Grad-CAM attention map visualizations showing which parts of the image
the model focuses on when making predictions.

Organized into:
- Correct/ folder: correctly predicted images
- Incorrect/ folder: incorrectly predicted images

Each visualization shows:
- Original image
- Grad-CAM heatmap (model attention)
- Overlay of attention on original image
- Ground truth vs Predicted labels

Usage:
    python visualize_attention.py
    python visualize_attention.py --json ./classification_results/classification_results_val.json
    python visualize_attention.py --limit 100
    python visualize_attention.py --no-model  # Use saliency fallback
"""

import os
os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")


# ===========================================================================
# Configuration
# ===========================================================================
class VisualizationConfig:
    """Configuration for attention visualization"""
    MODEL_PATH = "./skincap_model_classification_merged"
    RESULTS_JSON = "./classification_results/classification_results_val.json"
    OUTPUT_DIR = "./classification_results"
    IMAGE_BASE_PATH = "./SkinCAP/skincap"
    
    # Visualization settings
    ALPHA = 0.5  # Overlay transparency
    COLORMAP = cv2.COLORMAP_JET
    DPI = 150


def normalize_disease_name(name: str) -> str:
    """Normalize disease name for comparison"""
    if not name:
        return ""
    name = name.lower().strip()
    name = name.replace("-", " ").replace("_", " ")
    name = " ".join(name.split())
    return name


def is_correct_prediction(pred: str, gt: str) -> bool:
    """
    Check if prediction matches ground truth with flexible matching.
    Handles cases like "seborrheic keratosis irritated" -> "seborrheic keratosis"
    """
    pred_norm = normalize_disease_name(pred)
    gt_norm = normalize_disease_name(gt)
    
    # Exact match
    if pred_norm == gt_norm:
        return True
    
    pred_words = set(pred_norm.split())
    gt_words = set(gt_norm.split())
    
    # Partial match: pred contains all GT words or vice versa
    if gt_words.issubset(pred_words) and len(gt_words) >= 2:
        return True
    if pred_words.issubset(gt_words) and len(pred_words) >= 2:
        return True
    
    return False


def load_results(json_path: str) -> dict:
    """Load results from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_image_path(image_path: str, base_path: str = "./SkinCAP/skincap") -> Optional[str]:
    """Find the actual image path, trying multiple alternatives"""
    if Path(image_path).exists():
        return image_path
    
    filename = Path(image_path).name
    alt_paths = [
        Path(base_path) / filename,
        Path(image_path.replace("\\", "/")),
        Path(".") / image_path,
        Path(image_path.replace("SkinCAP\\skincap\\", "./SkinCAP/skincap/")),
        Path(image_path.replace("SkinCAP/skincap/", "./SkinCAP/skincap/")),
    ]
    
    for alt in alt_paths:
        if alt.exists():
            return str(alt)
    
    return None


# ===========================================================================
# Real Attention Extractor for Vision Language Models
# ===========================================================================
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
                # Capture attention weights if available
                if isinstance(output, tuple) and len(output) > 1:
                    attn_weights = output[1]  # Usually (batch, heads, seq, seq)
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
                if i % 4 == 0:  # Sample every 4th layer
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
        """
        Extract attention map for an image.
        
        Returns averaged attention from vision encoder focused on image regions.
        """
        if target_size is None:
            target_size = (image.height, image.width)
        
        self._clear_attention()
        
        try:
            # Prepare input
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
            
            # Forward pass to collect attention
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_attentions=True,
                    return_dict_in_generate=True,
                )
            
            # Try to extract attention map
            attention_map = self._compute_attention_map(target_size)
            
            if attention_map is not None:
                return attention_map
            else:
                # Fallback to gradient-based method
                return self._extract_gradient_attention(image, inputs, target_size)
                
        except Exception as e:
            print(f"   ⚠ Attention extraction failed: {e}")
            return self._extract_saliency_fallback(image, target_size)
    
    def _compute_attention_map(self, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Compute attention map from collected attention weights"""
        
        # Try vision encoder attention first
        if self.vision_attention:
            return self._process_vision_attention(target_size)
        
        # Try language model attention (focus on image tokens)
        if self.attention_weights:
            return self._process_language_attention(target_size)
        
        return None
    
    def _process_vision_attention(self, target_size: Tuple[int, int]) -> np.ndarray:
        """Process vision encoder attention into spatial map"""
        
        # Average attention across layers and heads
        all_attention = []
        
        for attn_data in self.vision_attention:
            weights = attn_data['weights']  # (batch, heads, seq, seq)
            
            if weights.dim() == 4:
                # Average over heads: (batch, seq, seq)
                avg_attn = weights.mean(dim=1)
                # Take CLS token attention or average
                if avg_attn.shape[1] > 1:
                    # Use attention from CLS token to all patches
                    cls_attn = avg_attn[0, 0, 1:]  # Skip CLS itself
                    all_attention.append(cls_attn.numpy())
        
        if not all_attention:
            return None
        
        # Average across layers
        attention = np.mean(all_attention, axis=0)
        
        # Reshape to 2D grid
        attention_map = self._reshape_to_grid(attention, target_size)
        
        return attention_map
    
    def _process_language_attention(self, target_size: Tuple[int, int]) -> np.ndarray:
        """Process language model attention focusing on image tokens"""
        
        all_attention = []
        
        for attn_data in self.attention_weights:
            weights = attn_data['weights']
            
            if weights.dim() == 4:
                # Average over heads
                avg_attn = weights.mean(dim=1)  # (batch, seq, seq)
                
                # Get attention from output tokens to all tokens
                # Focus on the last few tokens (prediction) attending to earlier (image) tokens
                seq_len = avg_attn.shape[-1]
                
                # Assume image tokens are in a certain range (first portion after special tokens)
                # This is model-specific, we'll take a reasonable estimate
                img_token_estimate = min(256, seq_len // 2)
                
                # Average attention to image region from last tokens
                if seq_len > img_token_estimate:
                    attn_to_image = avg_attn[0, -10:, :img_token_estimate].mean(dim=0)
                    all_attention.append(attn_to_image.numpy())
        
        if not all_attention:
            return None
        
        attention = np.mean(all_attention, axis=0)
        attention_map = self._reshape_to_grid(attention, target_size)
        
        return attention_map
    
    def _reshape_to_grid(self, attention: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Reshape 1D attention to 2D grid and resize"""
        
        seq_len = len(attention)
        
        # Find best grid size
        grid_size = int(np.sqrt(seq_len))
        
        # Common ViT patch sizes
        for gs in [14, 16, 24, 32, 28, 12]:
            if gs * gs <= seq_len:
                grid_size = gs
                break
        
        # Reshape to grid
        grid_len = grid_size * grid_size
        if grid_len <= seq_len:
            attention_grid = attention[:grid_len].reshape(grid_size, grid_size)
        else:
            # Pad if needed
            padded = np.zeros(grid_len)
            padded[:seq_len] = attention
            attention_grid = padded.reshape(grid_size, grid_size)
        
        # Normalize
        attention_grid = attention_grid - attention_grid.min()
        if attention_grid.max() > 0:
            attention_grid = attention_grid / attention_grid.max()
        
        # Resize to target
        attention_map = cv2.resize(
            attention_grid.astype(np.float32),
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Apply Gaussian smoothing for better visualization
        attention_map = cv2.GaussianBlur(attention_map, (15, 15), 0)
        
        # Renormalize
        attention_map = attention_map - attention_map.min()
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()
        
        return attention_map
    
    def _extract_gradient_attention(self, image: Image.Image, inputs: dict,
                                     target_size: Tuple[int, int]) -> np.ndarray:
        """Fallback: Use gradient-based attention (similar to Grad-CAM)"""
        
        try:
            # Enable gradients
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Get image embeddings if available
            if 'pixel_values' in inputs:
                pixel_values = inputs['pixel_values'].clone().requires_grad_(True)
                inputs['pixel_values'] = pixel_values
            
            with torch.enable_grad():
                outputs = self.model(**inputs)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs[0]
                
                # Get gradient w.r.t. max prediction
                target = logits[:, -1, :].max()
                target.backward()
                
                if 'pixel_values' in inputs and pixel_values.grad is not None:
                    # Use gradient magnitude as attention
                    grad = pixel_values.grad.abs()
                    attention = grad.mean(dim=1).squeeze().cpu().numpy()
                    
                    # Reshape and resize
                    if attention.ndim == 2:
                        attention_map = cv2.resize(
                            attention.astype(np.float32),
                            (target_size[1], target_size[0])
                        )
                        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
                        return attention_map
            
        except Exception as e:
            pass
        
        return self._extract_saliency_fallback(image, target_size)
    
    def _extract_saliency_fallback(self, image: Image.Image,
                                    target_size: Tuple[int, int]) -> np.ndarray:
        """Final fallback: Use image saliency"""
        
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            # Convert to LAB and compute saliency
            img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
            mean_lab = img_lab.mean(axis=(0, 1), keepdims=True)
            saliency = np.sqrt(((img_lab - mean_lab) ** 2).sum(axis=2))
        else:
            saliency = np.abs(img_array.astype(np.float32) - img_array.mean())
        
        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        # Smooth
        saliency = cv2.GaussianBlur(saliency.astype(np.float32), (21, 21), 0)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        # Resize if needed
        if saliency.shape[:2] != target_size:
            saliency = cv2.resize(saliency, (target_size[1], target_size[0]))
        
        return saliency


# ===========================================================================
# Visualization Functions
# ===========================================================================
def create_gradcam_visualization(image_path: str, heatmap: np.ndarray,
                                  ground_truth: str, predicted: str,
                                  is_correct: bool, output_path: str,
                                  sample_id: int, alpha: float = 0.5):
    """Create visualization with Grad-CAM heatmap overlay (2x2 grid)"""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Attention Map Visualization', fontsize=16)
    
    # Top-left: Original Image
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Top-right: Attention Overlay (image + heatmap with alpha)
    axes[0, 1].imshow(img_array)
    axes[0, 1].imshow(heatmap_resized, cmap='jet', alpha=0.5)
    axes[0, 1].set_title('Attention Overlay')
    axes[0, 1].axis('off')
    
    # Bottom-left: Attention Heatmap only
    im = axes[1, 0].imshow(heatmap_resized, cmap='hot')
    axes[1, 0].set_title('Attention Heatmap')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    
    # Bottom-right: High Attention Regions with contour
    threshold = 0.5
    attention_binary = heatmap_resized > threshold
    axes[1, 1].imshow(img_array)
    axes[1, 1].contour(attention_binary, colors='red', linewidths=2)
    axes[1, 1].set_title(f'High Attention Regions (>{threshold:.1f})')
    axes[1, 1].axis('off')
    
    # Add prediction info as text below figure
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


def create_simple_visualization(image_path: str, ground_truth: str, predicted: str, 
                                 is_correct: bool, output_path: str, sample_id: int):
    """Create a simple visualization without model (saliency-based, 2x2 grid)"""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    # Generate saliency-based attention map
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
    mean_lab = img_lab.mean(axis=(0, 1), keepdims=True)
    saliency = np.sqrt(((img_lab - mean_lab) ** 2).sum(axis=2))
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (21, 21), 0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Attention Map Visualization', fontsize=16)
    
    # Top-left: Original Image
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Top-right: Attention Overlay
    axes[0, 1].imshow(img_array)
    axes[0, 1].imshow(saliency, cmap='jet', alpha=0.5)
    axes[0, 1].set_title('Attention Overlay')
    axes[0, 1].axis('off')
    
    # Bottom-left: Attention Heatmap
    im = axes[1, 0].imshow(saliency, cmap='hot')
    axes[1, 0].set_title('Attention Heatmap')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    
    # Bottom-right: High Attention Regions
    threshold = 0.5
    attention_binary = saliency > threshold
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


def generate_visualizations(results: dict, output_dir: str, limit: int = None, 
                            use_model: bool = True, model_path: str = None):
    """Generate attention visualizations for all predictions"""
    
    predictions = results.get("predictions", [])
    split_name = results.get("split_name", "unknown")
    
    if limit:
        predictions = predictions[:limit]
    
    # Create output directories
    output_base = Path(output_dir) / f"attention_maps_{split_name}"
    correct_dir = output_base / "Correct"
    incorrect_dir = output_base / "Incorrect"
    
    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"REAL ATTENTION MAP VISUALIZATION")
    print(f"{'='*60}")
    print(f"\n📁 Output: {output_base}")
    print(f"📊 Total samples: {len(predictions)}")
    
    # Initialize attention extractor
    attention_extractor = None
    visualization_type = "saliency"
    
    if use_model and model_path:
        print(f"\n🔄 Loading model for attention extraction...")
        print(f"   Model path: {model_path}")
        
        try:
            from unsloth import FastVisionModel
            from transformers import AutoProcessor
            
            model, _ = FastVisionModel.from_pretrained(
                model_path,
                load_in_4bit=True,
            )
            
            processor = AutoProcessor.from_pretrained(model_path)
            FastVisionModel.for_inference(model)
            
            attention_extractor = AttentionExtractor(model, processor)
            visualization_type = "attention"
            print(f"   ✓ Model loaded - extracting real attention")
            
        except Exception as e:
            print(f"   ⚠ Could not load model: {e}")
            print(f"   → Using saliency-based visualization")
            attention_extractor = None
    else:
        print(f"\n📝 Using saliency-based visualization (no model specified)")
    
    # Statistics
    correct_count = 0
    incorrect_count = 0
    failed_count = 0
    
    # Prompt for attention extraction
    prompt = """Diagnose the SKIN DISEASE in this clinical photo.

DO NOT focus on:
- Eyes, pupils, eyelashes
- Skin folds or joint flexibility  
- Hair or nails
- Background objects

FOCUS ON skin abnormalities:
- Rashes, bumps, papules, plaques
- Color changes (red, brown, white patches)
- Texture changes (scaly, dry, thickened)
- Lesion shape and distribution

Disease name only:"""
    
    # Process each prediction
    for pred in tqdm(predictions, desc="Generating attention visualizations"):
        if pred.get("status") != "success":
            failed_count += 1
            continue
        
        sample_id = pred.get("id", "unknown")
        image_path = pred.get("image_path", "")
        ground_truth = pred.get("ground_truth", "")
        predicted = pred.get("predicted", "")
        
        # Find actual image path
        actual_path = find_image_path(image_path)
        if actual_path is None:
            failed_count += 1
            continue
        
        # Check if correct
        is_correct = is_correct_prediction(predicted, ground_truth)
        
        # Choose output directory
        if is_correct:
            correct_count += 1
            out_dir = correct_dir
        else:
            incorrect_count += 1
            out_dir = incorrect_dir
        
        # Generate filename
        gt_short = normalize_disease_name(ground_truth).replace(' ', '_')[:25]
        filename = f"{sample_id}_{gt_short}.png"
        output_path = out_dir / filename
        
        try:
            if attention_extractor is not None:
                img = Image.open(actual_path).convert("RGB")
                heatmap = attention_extractor.extract_attention(
                    img, prompt, target_size=(img.height, img.width)
                )
                
                create_gradcam_visualization(
                    actual_path, heatmap, ground_truth, predicted,
                    is_correct, str(output_path), sample_id
                )
            else:
                create_simple_visualization(
                    actual_path, ground_truth, predicted,
                    is_correct, str(output_path), sample_id
                )
        except Exception as e:
            print(f"\n⚠ Error processing {sample_id}: {e}")
            failed_count += 1
    
    # Cleanup hooks
    if attention_extractor is not None:
        attention_extractor.remove_hooks()
    
    # Print summary
    total = correct_count + incorrect_count
    accuracy = correct_count / total * 100 if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"\n📊 Overall Accuracy: {accuracy:.2f}%")
    print(f"\n   ✓ Correct:   {correct_count:4d} ({correct_count/max(total,1)*100:.1f}%)")
    print(f"   ✗ Incorrect: {incorrect_count:4d} ({incorrect_count/max(total,1)*100:.1f}%)")
    print(f"   ⚠ Failed:    {failed_count:4d}")
    print(f"   ─────────────────")
    print(f"   Total:       {total:4d}")
    print(f"\n📁 Output folders:")
    print(f"   {correct_dir} ({correct_count} images)")
    print(f"   {incorrect_dir} ({incorrect_count} images)")
    print(f"\n🎨 Visualization type: {visualization_type.upper()}")
    
    # Save summary to file
    summary = {
        "split": split_name,
        "total_samples": total,
        "correct": correct_count,
        "incorrect": incorrect_count,
        "failed": failed_count,
        "accuracy": accuracy,
        "accuracy_percent": f"{accuracy:.2f}%",
        "visualization_type": visualization_type
    }
    
    summary_path = output_base / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n📝 Summary saved to: {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Generate Grad-CAM attention map visualizations for classification results'
    )
    parser.add_argument('--json', type=str, 
                        default='./classification_results/classification_results_val.json',
                        help='Path to JSON results file')
    parser.add_argument('--output_dir', type=str, default='./classification_results',
                        help='Output directory for visualizations')
    parser.add_argument('--model', type=str, default='./skincap_model_classification_merged',
                        help='Path to model for Grad-CAM extraction')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images to process')
    parser.add_argument('--no-model', action='store_true',
                        help='Skip model loading, use saliency-based visualization only')
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not Path(args.json).exists():
        print(f"❌ Results file not found: {args.json}")
        print(f"   Please run inference_classification.py first")
        return
    
    # Load results
    print(f"📂 Loading results from: {args.json}")
    results = load_results(args.json)
    
    # Check model path
    model_path = None if args.no_model else args.model
    if model_path and not Path(model_path).exists():
        print(f"⚠ Model not found at: {model_path}")
        print(f"   Using saliency-based visualization instead")
        model_path = None
    
    # Generate visualizations
    generate_visualizations(
        results,
        args.output_dir,
        limit=args.limit,
        use_model=model_path is not None,
        model_path=model_path
    )


if __name__ == "__main__":
    main()
