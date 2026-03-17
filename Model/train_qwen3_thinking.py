"""
Training pipeline for Qwen3-VL-8B-Thinking using unsloth
Dataset: SkinCAP (local CSV file)
FORCES THINKING FORMAT in responses

This version adds <think> tags to training data
so the model learns to think before answering.
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


# Instruction that encourages thinking
INSTRUCTION = """Analyze this skin lesion image and provide a detailed clinical description.

Think through your analysis step by step before giving your final answer."""


def create_thinking_caption(caption: str, disease: str) -> str:
    """
    Wrap the original caption with thinking format.
    This teaches the model to reason before answering.
    """
    thinking_response = f"""<think>
Let me analyze this skin lesion image carefully.

First, I'll examine the visual features:
- Location and distribution of the lesion
- Color characteristics (pigmentation, erythema, etc.)
- Border definition (regular vs irregular)
- Surface texture and morphology
- Size and shape

Based on these clinical observations, I can see features consistent with {disease}.

Now I'll formulate a comprehensive clinical description.
</think>

{caption}"""
    
    return thinking_response


def convert_to_conversation(sample):
    """Convert sample to Qwen3-VL conversation format with thinking."""
    
    # Create thinking-enhanced caption
    thinking_caption = create_thinking_caption(
        caption=sample["caption"],
        disease=sample["disease"]
    )
    
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
                {"type": "text", "text": thinking_caption},
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
    
    df = df.dropna(subset=["skincap_file_path", "caption_zh_polish_en", "disease"])
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
                    "disease": row["disease"],  # เพิ่ม disease สำหรับ thinking
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
    
    # Load model and tokenizer
    print("Loading Qwen3-VL-8B-Thinking model...")
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
    
    # Convert to conversation format (with thinking)
    print("Converting to conversation format with THINKING...")
    dataset = dataset.map(
        convert_to_conversation,
        remove_columns=["image", "caption", "disease"],
        desc="Formatting dataset with thinking",
    )
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['test'])}")
    
    # Training configuration
    training_args = SFTConfig(
        output_dir="./qwen3_vl_8b_thinking_skincap",
        num_train_epochs=3,  # 3 epochs recommended
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
    
    # Initialize trainer
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
    print("Starting training with THINKING format...")
    print("="*50)
    print("Model will learn to:")
    print("1. Analyze image features")
    print("2. Think through diagnosis")
    print("3. Provide detailed response")
    print("="*50)
    trainer.train()
    
    # Save final model
    print("Saving model...")
    model.save_pretrained("./qwen3_vl_8b_thinking_skincap_final")
    tokenizer.save_pretrained("./qwen3_vl_8b_thinking_skincap_final")
    
    # Also save merged 16-bit model for inference
    print("Saving merged model for inference...")
    model.save_pretrained_merged(
        "./qwen3_vl_8b_thinking_skincap_merged",
        tokenizer,
        save_method="merged_16bit",
    )
    
    print("="*50)
    print("Training completed!")
    print("Model now trained to THINK before answering!")
    print("="*50)


if __name__ == "__main__":
    train()