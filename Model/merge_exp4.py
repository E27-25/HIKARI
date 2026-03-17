"""Merge Exp 4 LoRA checkpoint into full 16-bit model."""
import os
os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"

from unsloth import FastVisionModel

LORA_PATH   = "./skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_sts_classification"
MERGED_PATH = "./skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_sts_classification_merged"

print(f"Loading LoRA checkpoint: {LORA_PATH}")
model, tokenizer = FastVisionModel.from_pretrained(
    LORA_PATH,
    load_in_4bit=True,
)

print(f"Merging into: {MERGED_PATH}")
model.save_pretrained_merged(
    MERGED_PATH,
    tokenizer,
    save_method="merged_16bit",
)
print("Done.")
