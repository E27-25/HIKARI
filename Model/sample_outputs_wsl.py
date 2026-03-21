"""
sample_outputs_wsl.py
=====================
Run 3 samples through SGLang FP8 and vLLM BnB-4bit.
Shows raw output for Stage 2 (disease classification) and Stage 3 (captioning).

Run from WSL:
    python3 /mnt/c/Users/usEr/Desktop/Project/HIKARI/Model/sample_outputs_wsl.py
"""

import os, json, warnings
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda-12.8")
os.environ["SGLANG_DISABLE_DEEPGEMM"] = "1"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
warnings.filterwarnings("ignore")

from pathlib import Path
from PIL import Image
from transformers import AutoProcessor

BASE_DIR    = Path("/mnt/c/Users/usEr/Desktop/Project/HIKARI/Model")
STAGE2_MODEL = str(BASE_DIR / "skincap_fuzzytopk_s1cascade_ragR2_a09_classification_merged")
STAGE3_MODEL = str(BASE_DIR / "skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_classification_merged")

STAGE2_PROMPT_WITH_GROUP = (
    "This skin lesion belongs to the group '{group}'. Examine the lesion morphology "
    "(papules, plaques, macules), color (red, violet, white, brown), scale/crust, border "
    "sharpness, and distribution pattern. Based on these visual features, what is the "
    "specific skin disease?"
)
STAGE3_PROMPT = (
    "Describe this skin lesion image in detail. Include information about its "
    "appearance, possible diagnosis, and recommended examinations."
)

# ── pick 3 samples from existing val predictions ──────────────────────────────
PRED_FILE = BASE_DIR / "disease_classification_results" / \
            "results_disease_fuzzytopk_s1cascade_ragR2_a09_RAGR2_a09_val_predictions.json"

with open(PRED_FILE) as f:
    all_preds = json.load(f)["predictions"]

samples = all_preds[:3]   # first 3

def load_img(raw_path, size=672):
    p = BASE_DIR / raw_path.replace("\\", "/")
    img = Image.open(str(p)).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    return img

print("\n" + "="*70)
print("SAMPLES")
print("="*70)
for s in samples:
    print(f"  id={s['id']}  GT={s['ground_truth']}  group={s['disease_group']}")


# ══════════════════════════════════════════════════════════════════════════════
# SGLang FP8
# ══════════════════════════════════════════════════════════════════════════════
def run_sglang():
    import sglang as sgl
    print("\n" + "="*70)
    print("SGLang FP8")
    print("="*70)

    results = {}
    for stage, model_path, prompt_fn, max_tok in [
        (2, STAGE2_MODEL, None, 64),
        (3, STAGE3_MODEL, None, 256),
    ]:
        tag = f"Stage {stage}"
        print(f"\n  Loading {tag} model ...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        engine = sgl.Engine(
            model_path=model_path,
            dtype="bfloat16",
            quantization="fp8",
            context_length=2048,
            mem_fraction_static=0.88,
            trust_remote_code=True,
            disable_cuda_graph=True,
            log_level="error",
        )

        print(f"  Running {tag} on 3 samples ...")
        for s in samples:
            img = load_img(s["image_path"])
            group = s["disease_group"]

            if stage == 2:
                prompt_text = STAGE2_PROMPT_WITH_GROUP.format(group=group)
            else:
                prompt_text = STAGE3_PROMPT

            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            out = engine.generate(
                prompt=text,
                image_data=img,
                sampling_params={"max_new_tokens": max_tok, "temperature": 0.0},
            )
            raw = out["text"].strip() if isinstance(out, dict) else out[0]["text"].strip()
            results.setdefault(s["id"], {})[f"s{stage}_raw"] = raw

        engine.shutdown()
        import torch; torch.cuda.empty_cache()

    print("\n  ── RAW OUTPUTS ──")
    for s in samples:
        r = results[s["id"]]
        print(f"\n  id={s['id']}  GT={s['ground_truth']}")
        print(f"  [Stage 2] {repr(r['s2_raw'])}")
        print(f"  [Stage 3] {r['s3_raw'][:300]}...")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# vLLM BnB-4bit
# ══════════════════════════════════════════════════════════════════════════════
def run_vllm():
    from vllm import LLM, SamplingParams
    print("\n" + "="*70)
    print("vLLM BnB-4bit")
    print("="*70)

    results = {}
    for stage, model_path, max_tok in [
        (2, STAGE2_MODEL, 64),
        (3, STAGE3_MODEL, 256),
    ]:
        tag = f"Stage {stage}"
        print(f"\n  Loading {tag} model ...")
        llm = LLM(
            model=model_path,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.88,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        sp = SamplingParams(max_tokens=max_tok, temperature=0.0)

        print(f"  Running {tag} on 3 samples ...")
        for s in samples:
            img = load_img(s["image_path"])
            group = s["disease_group"]

            if stage == 2:
                prompt_text = STAGE2_PROMPT_WITH_GROUP.format(group=group)
            else:
                prompt_text = STAGE3_PROMPT

            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            import re
            img_count = text.count("<|vision_start|>")
            out = llm.generate(
                {"prompt": text, "multi_modal_data": {"image": [img] * max(img_count, 1)}},
                sp
            )
            raw = out[0].outputs[0].text.strip()
            results.setdefault(s["id"], {})[f"s{stage}_raw"] = raw

        del llm
        import torch, gc; torch.cuda.empty_cache(); gc.collect()

    print("\n  ── RAW OUTPUTS ──")
    for s in samples:
        r = results[s["id"]]
        print(f"\n  id={s['id']}  GT={s['ground_truth']}")
        print(f"  [Stage 2] {repr(r['s2_raw'])}")
        print(f"  [Stage 3] {r['s3_raw'][:300]}...")

    return results


if __name__ == "__main__":
    sgl_results  = run_sglang()
    vllm_results = run_vllm()

    # Save
    out = {"sglang_fp8": sgl_results, "vllm_bnb4": vllm_results}
    out_path = BASE_DIR / "disease_classification_results" / "results_sample_outputs.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")
