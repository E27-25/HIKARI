"""
bench_speed.py
==============
Measure Stage 2 (disease classification) and Stage 3 (caption generation)
inference throughput for two backends:

  Backend                  | Stage 2 ms/img | Stage 3 ms/img
  -------------------------|----------------|---------------
  Base (HF bfloat16)       |  mean +- std   |  mean +- std
  Unsloth 4-bit (single)   |  mean +- std   |  mean +- std
  Unsloth 4-bit (batch=4)  |  mean +- std   |  mean +- std

Notes
-----
- vLLM: no compiled Windows C extensions (no nvcc/CUDA toolkit on this host);
  vLLM deployment benchmarks require a Linux host with CUDA toolkit.
- LMDeploy TurboMind: same issue (needs nvcc to compile custom kernels).
- "Unsloth 4-bit batch=4" approximates vLLM-style continuous batching on
  the available platform.
- N_SAMPLES images drawn from the val set; N_RUNS timed passes after N_WARMUP
  warm-up passes; ms/img = pass_time / N_SAMPLES.

Usage
-----
    cd Model/
    python bench_speed.py
    python bench_speed.py --n_samples 10 --n_runs 5 --stage 2   # Stage 2 only
    python bench_speed.py --n_samples 10 --n_runs 5 --stage 3   # Stage 3 only
    python bench_speed.py --skip_hf                              # skip bfloat16 base
"""

import os, sys, json, time, argparse, statistics
from pathlib import Path

os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"]   = "1"

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "disease_classification_results"

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
STAGE2_MODEL = str(BASE_DIR / "skincap_fuzzytopk_s1cascade_ragR2_a09_classification_merged")
STAGE3_MODEL = str(BASE_DIR / "skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_classification_merged")

# Prompts (must match training)
DISEASE_PROMPT = (
    "What disease does this skin lesion image show? "
    "Respond with only the disease name."
)
CAPTION_PROMPT = (
    "Describe this skin lesion image in detail. Include information about its "
    "appearance, possible diagnosis, and recommended examinations."
)

# ---------------------------------------------------------------------------
# Load sample images from existing val predictions
# ---------------------------------------------------------------------------
def load_sample_images(n_samples: int, stage: int):
    pred_file = (
        RESULTS_DIR / "results_disease_fuzzytopk_s1cascade_RAGR2_a09_val_predictions.json"
        if stage == 2 else
        RESULTS_DIR / "results_caption_fuzzytopk_s1cascade_merged_init_stage3_val_predictions.json"
    )
    with open(pred_file) as f:
        preds = json.load(f)["predictions"]

    from PIL import Image
    images, ids = [], []
    for p in preds[:n_samples * 3]:  # try more in case some fail
        img_path = BASE_DIR / p["image_path"].replace("\\", "/")
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            ids.append(p["id"])
            if len(images) >= n_samples:
                break
        except Exception:
            continue
    return images, ids


# ---------------------------------------------------------------------------
# Backend 1 – Base: plain HuggingFace transformers (bfloat16, no quant)
# ---------------------------------------------------------------------------
def bench_hf_base(images, prompt: str, max_new_tokens: int,
                  n_warmup: int, n_runs: int):
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model_path = STAGE2_MODEL if "disease" in prompt.lower() or "What disease" in prompt else STAGE3_MODEL
    print(f"  [HF-Base] loading {Path(model_path).name} ...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()

    def _infer_one(img):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": prompt},
        ]}]
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, use_cache=True)
        return processor.decode(out[0], skip_special_tokens=True)

    # Warm-up
    print(f"    warm-up ({n_warmup} passes) ...")
    for _ in range(n_warmup):
        for img in images[:2]:
            _infer_one(img)

    # Timed runs
    times = []
    for r in range(n_runs):
        t0 = time.perf_counter()
        for img in images:
            _infer_one(img)
        elapsed = time.perf_counter() - t0
        ms_per_img = elapsed / len(images) * 1000
        times.append(ms_per_img)
        print(f"    run {r+1}/{n_runs}: {ms_per_img:.1f} ms/img")

    del model
    import torch; torch.cuda.empty_cache()
    return times


# ---------------------------------------------------------------------------
# Backend 2 – Unsloth 4-bit (current optimized approach)
# ---------------------------------------------------------------------------
def bench_unsloth(images, prompt: str, max_new_tokens: int,
                  n_warmup: int, n_runs: int, stage: int):
    import torch, re
    from unsloth import FastVisionModel

    model_path = STAGE2_MODEL if stage == 2 else STAGE3_MODEL
    print(f"  [Unsloth-4bit] loading {Path(model_path).name} ...")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=True,
        dtype=None,
    )
    FastVisionModel.for_inference(model)
    model.eval()

    def _infer_one(img):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": prompt},
        ]}]
        inputs = tokenizer.apply_chat_template(
            [messages], tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, padding=True,
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, use_cache=True,
                                 pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

    # Warm-up
    print(f"    warm-up ({n_warmup} passes) ...")
    for _ in range(n_warmup):
        for img in images[:2]:
            _infer_one(img)

    # Timed runs
    times = []
    for r in range(n_runs):
        t0 = time.perf_counter()
        for img in images:
            _infer_one(img)
        elapsed = time.perf_counter() - t0
        ms_per_img = elapsed / len(images) * 1000
        times.append(ms_per_img)
        print(f"    run {r+1}/{n_runs}: {ms_per_img:.1f} ms/img")

    del model
    import torch; torch.cuda.empty_cache()
    return times


# ---------------------------------------------------------------------------
# Backend 2b – Unsloth 4-bit batched (batch_size=4, approximates vLLM
#               continuous batching on Windows)
# ---------------------------------------------------------------------------
def bench_unsloth_batched(images, prompt: str, max_new_tokens: int,
                          n_warmup: int, n_runs: int, stage: int,
                          batch_size: int = 4):
    import torch, re
    from unsloth import FastVisionModel

    model_path = STAGE2_MODEL if stage == 2 else STAGE3_MODEL
    print(f"  [Unsloth-4bit-batch{batch_size}] loading {Path(model_path).name} ...")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_path, load_in_4bit=True, dtype=None,
    )
    FastVisionModel.for_inference(model)
    model.eval()

    def _infer_batch(imgs):
        batch_msgs = [[{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": prompt},
        ]}] for img in imgs]
        inputs = tokenizer.apply_chat_template(
            batch_msgs, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, padding=True,
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, use_cache=True,
                                 pad_token_id=tokenizer.pad_token_id)
        texts = [tokenizer.decode(o, skip_special_tokens=True) for o in out]
        return [re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
                for t in texts]

    # chunked batch helper
    def _run_all(imgs):
        for i in range(0, len(imgs), batch_size):
            _infer_batch(imgs[i: i + batch_size])

    # Warm-up
    print(f"    warm-up ({n_warmup} passes) ...")
    for _ in range(n_warmup):
        _run_all(images[:batch_size])

    # Timed runs
    times = []
    for r in range(n_runs):
        t0 = time.perf_counter()
        _run_all(images)
        elapsed = time.perf_counter() - t0
        ms_per_img = elapsed / len(images) * 1000
        times.append(ms_per_img)
        print(f"    run {r+1}/{n_runs}: {ms_per_img:.1f} ms/img")

    del model
    torch.cuda.empty_cache()
    return times


# ---------------------------------------------------------------------------
# Backend 3 – LMDeploy TurboMind (vLLM-equivalent: paged attention,
#              continuous batching, quantized kernel fusion)
# ---------------------------------------------------------------------------
def bench_lmdeploy(images, prompt: str, max_new_tokens: int,
                   n_warmup: int, n_runs: int, stage: int):
    import torch, re, tempfile, os
    from PIL import Image as PILImage

    model_path = STAGE2_MODEL if stage == 2 else STAGE3_MODEL
    print(f"  [LMDeploy-TurboMind] loading {Path(model_path).name} ...")

    try:
        from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
        from lmdeploy.vl import load_image

        engine_cfg = TurbomindEngineConfig(
            tp=1,
            max_batch_size=4,
            cache_max_entry_count=0.7,
        )
        pipe = pipeline(
            model_path,
            backend_config=engine_cfg,
        )
    except Exception as e:
        print(f"    [LMDeploy] TurboMind failed ({e}), trying PyTorch backend ...")
        try:
            from lmdeploy import pipeline, PytorchEngineConfig
            from lmdeploy.vl import load_image

            engine_cfg = PytorchEngineConfig(tp=1, max_batch_size=4)
            pipe = pipeline(model_path, backend_config=engine_cfg)
        except Exception as e2:
            print(f"    [LMDeploy] PyTorch backend also failed: {e2}")
            return None

    def _infer_one(img):
        # LMDeploy VL pipeline accepts image + text in messages
        response = pipe([(img, prompt)])
        return response[0].text

    # Warm-up
    print(f"    warm-up ({n_warmup} passes) ...")
    try:
        for _ in range(n_warmup):
            for img in images[:2]:
                _infer_one(img)
    except Exception as e:
        print(f"    [LMDeploy] warm-up failed: {e}")
        del pipe
        torch.cuda.empty_cache()
        return None

    # Timed runs
    times = []
    for r in range(n_runs):
        t0 = time.perf_counter()
        for img in images:
            _infer_one(img)
        elapsed = time.perf_counter() - t0
        ms_per_img = elapsed / len(images) * 1000
        times.append(ms_per_img)
        print(f"    run {r+1}/{n_runs}: {ms_per_img:.1f} ms/img")

    del pipe
    torch.cuda.empty_cache()
    return times


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------
def stats(times):
    if not times:
        return None, None
    mean = statistics.mean(times)
    std  = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Images per timed pass")
    parser.add_argument("--n_runs",    type=int, default=5,
                        help="Timed runs (for +/- std)")
    parser.add_argument("--n_warmup",  type=int, default=2,
                        help="Warm-up passes (not timed)")
    parser.add_argument("--stage",     type=int, default=0,
                        choices=[0, 2, 3],
                        help="0=both, 2=Stage2 only, 3=Stage3 only")
    parser.add_argument("--skip_hf",   action="store_true",
                        help="Skip HuggingFace base benchmark (slow)")
    parser.add_argument("--skip_lmd",  action="store_true",
                        help="Skip LMDeploy benchmark")
    args = parser.parse_args()

    stages = [2, 3] if args.stage == 0 else [args.stage]
    results = {}  # {(stage, backend): (mean, std)}

    for stage in stages:
        prompt = DISEASE_PROMPT if stage == 2 else CAPTION_PROMPT
        max_tok = 64 if stage == 2 else 256
        label   = f"Stage {stage} ({'Disease Classif.' if stage == 2 else 'Caption Gen.'})"

        print(f"\n{'='*70}")
        print(f"  Benchmarking {label}")
        print(f"  n_samples={args.n_samples}  n_runs={args.n_runs}  "
              f"n_warmup={args.n_warmup}")
        print(f"{'='*70}")

        print(f"\nLoading {args.n_samples} sample images ...")
        images, ids = load_sample_images(args.n_samples, stage)
        print(f"  Loaded {len(images)} images: {ids[:5]} ...")

        if len(images) < args.n_samples:
            print(f"  Warning: only {len(images)} images available (requested {args.n_samples})")

        # -- Base HF --
        if not args.skip_hf:
            print(f"\n[1/3] HuggingFace base (bfloat16, no quantization)")
            try:
                times = bench_hf_base(images, prompt, max_tok, args.n_warmup, args.n_runs)
                m, s  = stats(times)
                results[(stage, "HF-Base")] = (m, s)
                print(f"  => {m:.1f} +- {s:.1f} ms/img")
            except Exception as e:
                print(f"  HF-Base FAILED: {e}")
                results[(stage, "HF-Base")] = (None, None)

        # -- Unsloth 4-bit single --
        print(f"\n[2/4] Unsloth 4-bit single-sample (base inference)")
        try:
            times = bench_unsloth(images, prompt, max_tok, args.n_warmup, args.n_runs, stage)
            m, s  = stats(times)
            results[(stage, "Unsloth-4bit-single")] = (m, s)
            print(f"  => {m:.1f} +- {s:.1f} ms/img")
        except Exception as e:
            print(f"  Unsloth FAILED: {e}")
            results[(stage, "Unsloth-4bit-single")] = (None, None)

        # -- Unsloth 4-bit batched --
        print(f"\n[3/4] Unsloth 4-bit batch=4 (vLLM-style throughput)")
        try:
            times = bench_unsloth_batched(images, prompt, max_tok, args.n_warmup, args.n_runs, stage, batch_size=4)
            m, s  = stats(times)
            results[(stage, "Unsloth-4bit-batch4")] = (m, s)
            print(f"  => {m:.1f} +- {s:.1f} ms/img")
        except Exception as e:
            print(f"  Unsloth batched FAILED: {e}")
            results[(stage, "Unsloth-4bit-batch4")] = (None, None)

        # -- LMDeploy --
        if not args.skip_lmd:
            print(f"\n[4/4] LMDeploy (vLLM-equivalent: paged-attention + cont. batching)")
            try:
                times = bench_lmdeploy(images, prompt, max_tok, args.n_warmup, args.n_runs, stage)
                if times is not None:
                    m, s = stats(times)
                    results[(stage, "LMDeploy")] = (m, s)
                    print(f"  => {m:.1f} +- {s:.1f} ms/img")
                else:
                    results[(stage, "LMDeploy")] = (None, None)
            except Exception as e:
                print(f"  LMDeploy FAILED: {e}")
                results[(stage, "LMDeploy")] = (None, None)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    backends = ["HF-Base", "Unsloth-4bit-single", "Unsloth-4bit-batch4", "LMDeploy"]
    print("\n" + "="*76)
    print(f"{'Backend':<26}  {'Stage 2 ms/img':>20}  {'Stage 3 ms/img':>20}")
    print("-"*76)
    for backend in backends:
        s2 = results.get((2, backend))
        s3 = results.get((3, backend))
        def fmt(r):
            if r is None or r[0] is None:
                return f"{'N/A (see notes)':>20}"
            return f"{r[0]:>9.1f} +- {r[1]:>5.1f}"
        print(f"{backend:<26}  {fmt(s2)}  {fmt(s3)}")
    print("="*76)
    print("Notes: HF-Base=bfloat16 no quant; vLLM/LMDeploy require Linux+nvcc")

    # Save JSON
    out = {}
    for (stage, backend), (m, s) in results.items():
        out.setdefault(f"stage{stage}", {})[backend] = {
            "mean_ms_per_img": round(m, 2) if m else None,
            "std_ms_per_img":  round(s, 2) if s else None,
            "imgs_per_sec":    round(1000 / m, 2) if m else None,
        }
    out_path = RESULTS_DIR / "results_speed_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved -> {out_path}")


if __name__ == "__main__":
    main()
