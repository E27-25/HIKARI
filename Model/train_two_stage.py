"""
Two-Stage Training Pipeline for SkinCAP Dataset
Based on: OralGPT Paper (Zhang et al., arXiv:2510.13911v1)

Model: Qwen/Qwen3-VL-8B-Thinking
Framework: Unsloth

OUTPUT: 2 Models
- Model 1 (Stage 1): Classification - สำหรับวินิจฉัยโรค
- Model 2 (Stage 2): Caption - สำหรับอธิบายภาพ (train ต่อจาก Stage 1)
"""

import os
import random
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from datasets import Dataset
from typing import List, Dict, Optional

# Disable torch.compile
os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from unsloth import FastVisionModel, is_bf16_supported
from trl import SFTTrainer, SFTConfig


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Random seed (same as original)
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
    
    # Output directories - แยก 2 models
    OUTPUT_DIR_STAGE1 = "./skincap_model_classification"
    OUTPUT_DIR_STAGE1_MERGED = "./skincap_model_classification_merged"
    
    OUTPUT_DIR_STAGE2 = "./skincap_model_caption"
    OUTPUT_DIR_STAGE2_MERGED = "./skincap_model_caption_merged"
    
    # Data paths
    CSV_PATH = "./SkinCAP/skincap_v240623.csv"
    IMAGE_BASE_PATH = "./SkinCAP/skincap"


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
    print(f"✓ Seed set to {seed}")


def setup_environment():
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    set_seed(Config.SEED)
    print("✓ Environment configured")


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
# DATA CONVERSION
# ============================================================================

def convert_classification(sample):
    """Stage 1: Classification format - loads image lazily from path"""
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
                {"type": "text", "text": sample["answer"]},
            ],
        },
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
            # Store path as string, not the loaded image (saves memory)
            data_list.append({
                "image_path": str(img_path),
                "disease": row["disease"],
                "caption": row["caption_zh_polish_en"],
            })
    
    print(f"✓ Found {len(data_list)} valid samples")
    return data_list


def prepare_classification_data(data: List[Dict], num_prompts: int = 3) -> Dataset:
    """Prepare data for Stage 1: Classification"""
    print("Preparing classification dataset...")
    
    samples = []
    for item in data:
        selected_prompts = random.sample(CLASSIFICATION_PROMPTS, min(num_prompts, len(CLASSIFICATION_PROMPTS)))
        
        for prompt in selected_prompts:
            samples.append({
                "image_path": item["image_path"],
                "prompt": prompt,
                "answer": f"This image shows {item['disease']}.",
            })
    
    random.shuffle(samples)
    print(f"✓ Created {len(samples)} classification samples")
    return Dataset.from_list(samples)


def prepare_caption_data(data: List[Dict], num_prompts: int = 3) -> Dataset:
    """Prepare data for Stage 2: Caption"""
    print("Preparing caption dataset...")
    
    samples = []
    for item in data:
        selected_prompts = random.sample(CAPTION_PROMPTS, min(num_prompts, len(CAPTION_PROMPTS)))
        
        for prompt in selected_prompts:
            samples.append({
                "image_path": item["image_path"],
                "prompt": prompt,
                "caption": item["caption"],
            })
    
    random.shuffle(samples)
    print(f"✓ Created {len(samples)} caption samples")
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
    
    print(f"✓ Model loaded with LoRA (r={Config.LORA_R}, alpha={Config.LORA_ALPHA})")
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
    
    print(f"✓ Model loaded from checkpoint (LoRA adapters already present)")
    return model, tokenizer


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_stage1(train_data: List[Dict], val_data: List[Dict]):
    """
    Stage 1: Classification Training
    
    Output: Model ที่วินิจฉัยโรคได้
    """
    print("\n" + "="*60)
    print("STAGE 1: CLASSIFICATION TRAINING")
    print("="*60)
    
    # Load base model
    model, tokenizer = load_base_model()
    
    # Prepare data
    train_dataset = prepare_classification_data(train_data, num_prompts=3)
    val_dataset = prepare_classification_data(val_data, num_prompts=1)
    
    # Convert format
    train_dataset = train_dataset.map(
        convert_classification,
        remove_columns=["image_path", "prompt", "answer"],
        desc="Formatting train data",
    )
    val_dataset = val_dataset.map(
        convert_classification,
        remove_columns=["image_path", "prompt", "answer"],
        desc="Formatting val data",
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
    print("\n" + "="*60)
    print("💾 SAVING MODEL 1: CLASSIFICATION")
    print("="*60)
    
    model.save_pretrained(Config.OUTPUT_DIR_STAGE1)
    tokenizer.save_pretrained(Config.OUTPUT_DIR_STAGE1)
    print(f"✓ LoRA saved to: {Config.OUTPUT_DIR_STAGE1}")
    
    model.save_pretrained_merged(
        Config.OUTPUT_DIR_STAGE1_MERGED,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"✓ Merged model saved to: {Config.OUTPUT_DIR_STAGE1_MERGED}")
    
    return model, tokenizer


def train_stage2(train_data: List[Dict], val_data: List[Dict]):
    """
    Stage 2: Caption Training
    
    Loads from Stage 1 checkpoint
    Output: Model ที่อธิบายภาพได้
    """
    print("\n" + "="*60)
    print("STAGE 2: CAPTION TRAINING")
    print("="*60)
    
    # Load from Stage 1 checkpoint
    model, tokenizer = load_from_checkpoint(Config.OUTPUT_DIR_STAGE1)
    
    # Prepare data
    train_dataset = prepare_caption_data(train_data, num_prompts=3)
    val_dataset = prepare_caption_data(val_data, num_prompts=1)
    
    # Convert format
    train_dataset = train_dataset.map(
        convert_caption,
        remove_columns=["image_path", "prompt", "caption"],
        desc="Formatting train data",
    )
    val_dataset = val_dataset.map(
        convert_caption,
        remove_columns=["image_path", "prompt", "caption"],
        desc="Formatting val data",
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
    print("\n" + "="*60)
    print("💾 SAVING MODEL 2: CAPTION")
    print("="*60)
    
    model.save_pretrained(Config.OUTPUT_DIR_STAGE2)
    tokenizer.save_pretrained(Config.OUTPUT_DIR_STAGE2)
    print(f"✓ LoRA saved to: {Config.OUTPUT_DIR_STAGE2}")
    
    model.save_pretrained_merged(
        Config.OUTPUT_DIR_STAGE2_MERGED,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"✓ Merged model saved to: {Config.OUTPUT_DIR_STAGE2_MERGED}")
    
    return model, tokenizer


# ============================================================================
# MAIN
# ============================================================================

def train_two_stage():
    """Full two-stage training pipeline"""
    
    print("="*60)
    print("TWO-STAGE TRAINING PIPELINE")
    print("="*60)
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Seed: {Config.SEED}")
    print(f"Stage 1 epochs: {Config.STAGE1_EPOCHS}")
    print(f"Stage 2 epochs: {Config.STAGE2_EPOCHS}")
    print("="*60)
    
    setup_environment()
    
    # Load data
    print("\n[1/4] Loading data...")
    data = load_skincap_data()
    
    # Split 90/10
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Stage 1: Classification
    print("\n[2/4] Stage 1: Classification...")
    train_stage1(train_data, val_data)
    
    # Stage 2: Caption (loads from Stage 1)
    print("\n[3/4] Stage 2: Caption...")
    train_stage2(train_data, val_data)
    
    # Summary
    print("\n" + "="*60)
    print("[4/4] TRAINING COMPLETED!")
    print("="*60)
    print("\n📦 OUTPUT MODELS:")
    print(f"""
┌─────────────────────────────────────────────────────────────┐
│  MODEL 1: CLASSIFICATION (สำหรับวินิจฉัยโรค)                 │
│  ─────────────────────────────────────────                  │
│  LoRA:   {Config.OUTPUT_DIR_STAGE1:<40} │
│  Merged: {Config.OUTPUT_DIR_STAGE1_MERGED:<40} │
│                                                             │
│  Usage: ถามว่า "โรคอะไร?" → ตอบชื่อโรค                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  MODEL 2: CAPTION (สำหรับอธิบายภาพ)                          │
│  ─────────────────────────────────────────                  │
│  LoRA:   {Config.OUTPUT_DIR_STAGE2:<40} │
│  Merged: {Config.OUTPUT_DIR_STAGE2_MERGED:<40} │
│                                                             │
│  Usage: ถามว่า "อธิบายภาพ" → ตอบ caption ละเอียด              │
│                                                             │
│  Note: Model นี้ train ต่อจาก Model 1                        │
│        ทำ classification ได้ดีขึ้นด้วย (ตาม paper)            │
└─────────────────────────────────────────────────────────────┘
""")


def train_stage1_only():
    """Train only Stage 1"""
    setup_environment()
    data = load_skincap_data()
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    train_stage1(data[:split_idx], data[split_idx:])


def train_stage2_only():
    """Train only Stage 2 (requires Stage 1 checkpoint)"""
    if not Path(Config.OUTPUT_DIR_STAGE1).exists():
        print(f"Error: Stage 1 checkpoint not found at {Config.OUTPUT_DIR_STAGE1}")
        print("Please run Stage 1 first.")
        return
    
    setup_environment()
    data = load_skincap_data()
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    train_stage2(data[:split_idx], data[split_idx:])


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Two-Stage Training")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["both", "stage1", "stage2"],
                        help="Training mode")
    parser.add_argument("--stage1_epochs", type=int, default=3)
    parser.add_argument("--stage2_epochs", type=int, default=3)
    
    args = parser.parse_args()
    
    Config.STAGE1_EPOCHS = args.stage1_epochs
    Config.STAGE2_EPOCHS = args.stage2_epochs
    
    if args.mode == "both":
        train_two_stage()
    elif args.mode == "stage1":
        train_stage1_only()
    elif args.mode == "stage2":
        train_stage2_only()