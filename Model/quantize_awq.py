"""
quantize_awq.py
===============
Pre-quantize Stage 2 and Stage 3 models to 4-bit AWQ format so LMDeploy
can load them in already-quantized form (avoids OOM from fp16 intermediate load).

Run from WSL:
    python3 /mnt/c/Users/usEr/Desktop/Project/HIKARI/Model/quantize_awq.py
    python3 /mnt/c/Users/usEr/Desktop/Project/HIKARI/Model/quantize_awq.py --stage 2
    python3 /mnt/c/Users/usEr/Desktop/Project/HIKARI/Model/quantize_awq.py --stage 3
"""

import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.NewGELUActivation

import os, argparse
from pathlib import Path
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

BASE_DIR = Path("/mnt/c/Users/usEr/Desktop/Project/HIKARI/Model")

MODELS = {
    2: BASE_DIR / "skincap_fuzzytopk_s1cascade_ragR2_a09_classification_merged",
    3: BASE_DIR / "skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_classification_merged",
}
AWQ_MODELS = {
    2: BASE_DIR / "skincap_stage2_awq4bit",
    3: BASE_DIR / "skincap_stage3_awq4bit",
}

AWQ_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}


def quantize(stage):
    model_path = str(MODELS[stage])
    out_path   = str(AWQ_MODELS[stage])
    print(f"\n=== Stage {stage}: {Path(model_path).name} → {Path(out_path).name} ===")

    print("  Loading model in fp16 ...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("  Running AWQ calibration ...")
    model.quantize(tokenizer, quant_config=AWQ_CONFIG)

    print(f"  Saving to {out_path} ...")
    model.save_quantized(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"  Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[2, 3], default=None)
    args = parser.parse_args()

    stages = [args.stage] if args.stage else [2, 3]
    for s in stages:
        quantize(s)


if __name__ == "__main__":
    main()
