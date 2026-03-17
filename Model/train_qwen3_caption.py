"""
Training pipeline for Qwen3-VL-8B-Thinking using unsloth
Dataset: SkinCAP (local CSV file)
Uses: image and caption_zh_polish_en columns

This is the baseline SFT caption training.
Based on original train.py, upgraded to Qwen3-VL-8B-Thinking.
"""

import os
import pandas as pd
from pathlib import Path
from PIL import Image
from datasets import Dataset

# Disable torch.compile (requires C++ compiler on Windows)
os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from unsloth import FastVisionModel, is_bf16_supported
from trl import SFTTrainer, SFTConfig


def setup_environment():
    """Configure environment variables for efficient training."""
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    print("Environment configured for training")


# Instruction for skin lesion analysis
INSTRUCTION = "Describe this skin lesion image in detail. Include information about its appearance, possible diagnosis, and recommended examinations."


def convert_to_conversation(sample):
    """Convert sample to Qwen3-VL conversation format."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": INSTRUCTION},
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


def load_dataset_from_local():
    """Load SkinCAP dataset from local CSV file."""
    print("Loading SkinCAP dataset from local CSV...")
    
    csv_path = Path("./SkinCAP/skincap_v240623.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from CSV")
    
    df = df.dropna(subset=["skincap_file_path", "caption_zh_polish_en"])
    print(f"After filtering: {len(df)} valid samples")
    
    image_base_path = Path("./SkinCAP/skincap")
    
    # Prepare data with actual PIL images
    data_list = []
    for idx, row in df.iterrows():
        img_path = image_base_path / row["skincap_file_path"]
        
        if img_path.exists():
            try:
                # Load image as PIL
                image = Image.open(str(img_path)).convert("RGB")
                data_list.append({
                    "image": image,
                    "caption": row["caption_zh_polish_en"],
                })
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
    
    print(f"Found {len(data_list)} valid image-caption pairs")
    
    # Create dataset
    dataset = Dataset.from_list(data_list)
    
    return dataset


def train():
    """Main training function."""
    setup_environment()
    
    # Load model and tokenizer - using Qwen3-VL-8B-Thinking
    print("Loading model with unsloth optimization...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "Qwen/Qwen3-VL-8B-Thinking",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    print("Model loaded successfully!")
    
    # Prepare model for LoRA training
    print("Preparing model for training with LoRA...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    print("Model prepared for training!")
    
    # Load dataset
    dataset = load_dataset_from_local()
    
    # Convert to conversation format
    print("Converting to conversation format...")
    dataset = dataset.map(
        convert_to_conversation,
        remove_columns=["image", "caption"],
        desc="Formatting dataset",
    )
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['test'])}")
    
    # Training configuration
    training_args = SFTConfig(
        output_dir="./qwen3_vl_8b_skincap",
        num_train_epochs=1,  # Start with 1 epoch to test
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=2048,
    )
    
    # Initialize trainer - use unsloth's built-in vision support
    print("Initializing trainer...")
    from unsloth import UnslothVisionDataCollator
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving model...")
    model.save_pretrained("./qwen3_vl_8b_skincap_final")
    tokenizer.save_pretrained("./qwen3_vl_8b_skincap_final")
    
    # Also save merged 16-bit model for inference
    print("Saving merged model for inference...")
    model.save_pretrained_merged(
        "./qwen3_vl_8b_skincap_merged",
        tokenizer,
        save_method="merged_16bit",
    )
    
    print("Training completed!")


if __name__ == "__main__":
    train()