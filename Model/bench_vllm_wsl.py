"""
bench_vllm_wsl.py
=================
Benchmark Stage 2 and Stage 3 inference throughput using vLLM (WSL2/Linux).
Run from WSL:
    python3 /mnt/c/Users/usEr/Desktop/Project/HIKARI/Model/bench_vllm_wsl.py

Compares:
  - Unsloth 4-bit single   (base)
  - vLLM BnB-4bit single   (vLLM no-batch)
  - vLLM BnB-4bit batch=4  (vLLM batch, main speedup)
"""

import os, sys, json, time, statistics
from pathlib import Path

BASE_DIR    = Path("/mnt/c/Users/usEr/Desktop/Project/HIKARI/Model")
RESULTS_DIR = BASE_DIR / "disease_classification_results"

STAGE2_MODEL = str(BASE_DIR / "skincap_fuzzytopk_s1cascade_ragR2_a09_classification_merged")
STAGE3_MODEL = str(BASE_DIR / "skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_classification_merged")

DISEASE_PROMPT = (
    "What disease does this skin lesion image show? "
    "Respond with only the disease name."
)
CAPTION_PROMPT = (
    "Describe this skin lesion image in detail. Include information about its "
    "appearance, possible diagnosis, and recommended examinations."
)

N_SAMPLES = 8
N_WARMUP  = 1
N_RUNS    = 3


def load_images(n, stage):
    pred_file = (
        RESULTS_DIR / "results_disease_fuzzytopk_s1cascade_RAGR2_a09_val_predictions.json"
        if stage == 2 else
        RESULTS_DIR / "results_caption_fuzzytopk_s1cascade_merged_init_stage3_val_predictions.json"
    )
    with open(pred_file) as f:
        preds = json.load(f)["predictions"]
    from PIL import Image
    imgs = []
    for p in preds[:n * 3]:
        try:
            img = Image.open(BASE_DIR / p["image_path"].replace("\\", "/")).convert("RGB")
            imgs.append(img)
            if len(imgs) >= n:
                break
        except Exception:
            continue
    return imgs


def stats(times):
    if not times:
        return None, None
    return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0)


# ---------------------------------------------------------------------------
# vLLM benchmark
# ---------------------------------------------------------------------------
def bench_vllm(images, prompt, max_new_tokens, stage, batch_size=1):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TextPrompt

    model_path = STAGE2_MODEL if stage == 2 else STAGE3_MODEL
    tag = f"vLLM-BnB4-bs{batch_size}"
    print(f"\n  [{tag}] loading {Path(model_path).name} ...")

    llm = LLM(
        model=model_path,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        dtype="bfloat16",
        max_model_len=1024 if stage == 2 else 2048,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
    )

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path)

    def build_inputs(imgs):
        batch = []
        for img in imgs:
            # vLLM native multimodal: image placeholder only in messages,
            # actual PIL image passed separately in multi_modal_data
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch.append({
                "prompt": text,
                "multi_modal_data": {"image": img},
            })
        return batch

    def run_batch(imgs):
        inputs = build_inputs(imgs)
        llm.generate(inputs, sampling_params=sampling)

    def run_all():
        for i in range(0, len(images), batch_size):
            run_batch(images[i: i + batch_size])

    # warm-up
    print(f"    warm-up ({N_WARMUP}) ...")
    for _ in range(N_WARMUP):
        run_batch(images[:batch_size])

    # timed runs
    times = []
    for r in range(N_RUNS):
        t0 = time.perf_counter()
        run_all()
        elapsed = time.perf_counter() - t0
        ms = elapsed / len(images) * 1000
        times.append(ms)
        print(f"    run {r+1}/{N_RUNS}: {ms:.1f} ms/img")

    del llm
    import torch; torch.cuda.empty_cache()
    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    results = {}
    for stage in [2, 3]:
        prompt   = DISEASE_PROMPT if stage == 2 else CAPTION_PROMPT
        max_tok  = 64 if stage == 2 else 256
        label    = f"Stage {stage}"

        print(f"\n{'='*60}")
        print(f"  {label}  n_samples={N_SAMPLES}  n_runs={N_RUNS}")
        print(f"{'='*60}")

        images = load_images(N_SAMPLES, stage)
        print(f"  Loaded {len(images)} images")

        # vLLM single
        try:
            t = bench_vllm(images, prompt, max_tok, stage, batch_size=1)
            m, s = stats(t)
            results[(stage, "vLLM-BnB4-bs1")] = (m, s)
            print(f"  => vLLM single: {m:.1f} +- {s:.1f} ms/img")
        except Exception as e:
            print(f"  vLLM single FAILED: {e}")
            results[(stage, "vLLM-BnB4-bs1")] = (None, None)

        # vLLM batch=4
        try:
            t = bench_vllm(images, prompt, max_tok, stage, batch_size=4)
            m, s = stats(t)
            results[(stage, "vLLM-BnB4-bs4")] = (m, s)
            print(f"  => vLLM batch=4: {m:.1f} +- {s:.1f} ms/img")
        except Exception as e:
            print(f"  vLLM batch=4 FAILED: {e}")
            results[(stage, "vLLM-BnB4-bs4")] = (None, None)

    # Print summary
    print("\n" + "="*64)
    print(f"{'Backend':<22}  {'Stage 2 ms/img':>20}  {'Stage 3 ms/img':>20}")
    print("-"*64)
    for backend in ["vLLM-BnB4-bs1", "vLLM-BnB4-bs4"]:
        s2 = results.get((2, backend), (None, None))
        s3 = results.get((3, backend), (None, None))
        def fmt(r):
            if r is None or r[0] is None:
                return f"{'N/A':>20}"
            return f"{r[0]:>9.1f} +- {r[1]:>5.1f}"
        print(f"{backend:<22}  {fmt(s2)}  {fmt(s3)}")
    print("="*64)

    # Save
    out = {}
    for (stage, backend), (m, s) in results.items():
        out.setdefault(f"stage{stage}", {})[backend] = {
            "mean_ms_per_img": round(m, 2) if m else None,
            "std_ms_per_img":  round(s, 2) if s else None,
            "imgs_per_sec":    round(1000/m, 2) if m else None,
        }
    out_path = RESULTS_DIR / "results_speed_vllm.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
