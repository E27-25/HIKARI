"""
bench_engines_wsl.py
====================
Benchmark Stage 2 and Stage 3 inference throughput across engines:
  - SGLang  (offline batch engine)
  - LMDeploy (PyTorch backend, TurboMind if available)

Run from WSL:
    python3 /mnt/c/Users/usEr/Desktop/Project/HIKARI/Model/bench_engines_wsl.py
    python3 /mnt/c/Users/usEr/Desktop/Project/HIKARI/Model/bench_engines_wsl.py --engine sglang
    python3 /mnt/c/Users/usEr/Desktop/Project/HIKARI/Model/bench_engines_wsl.py --engine lmdeploy
"""

import os, sys, json, time, statistics, argparse
from pathlib import Path

# CUDA 12.8 toolkit path (for sgl_kernel FA3 preload and nvcc)
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda-12.8")
# Disable deep_gemm JIT (requires nvcc/CUDA toolkit not available in WSL2)
os.environ["SGLANG_DISABLE_DEEPGEMM"] = "1"
os.environ["SGL_DISABLE_DEEPGEMM"] = "1"
# Bypass flashinfer version mismatch (cubin 0.5.2 vs flashinfer 0.6.6)
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

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
N_WARMUP  = 3  # SGLang JIT-compiles FlashInfer kernels on first requests; 3 warmups ensures steady-state
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
    imgs, paths = [], []
    for p in preds[:n * 3]:
        try:
            path = str(BASE_DIR / p["image_path"].replace("\\", "/"))
            img = Image.open(path).convert("RGB")
            imgs.append(img)
            paths.append(path)
            if len(imgs) >= n:
                break
        except Exception:
            continue
    return imgs, paths


def stats(times):
    if not times:
        return None, None
    return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0)


# ---------------------------------------------------------------------------
# SGLang benchmark
# ---------------------------------------------------------------------------
def bench_sglang(images, image_paths, prompt, max_new_tokens, stage, batch_size=1):
    import sglang as sgl
    from transformers import AutoProcessor

    model_path = STAGE2_MODEL if stage == 2 else STAGE3_MODEL
    tag = f"SGLang-bs{batch_size}"
    print(f"\n  [{tag}] loading {Path(model_path).name} ...")

    # SGLang offline engine with FP8 quantization to fit 17GB VRAM
    # (SGLang 0.5.5.post3 does not support bitsandbytes; fp8 reduces weights ~50%)
    engine = sgl.Engine(
        model_path=model_path,
        dtype="bfloat16",
        quantization="fp8",
        context_length=1024 if stage == 2 else 2048,
        mem_fraction_static=0.88,
        trust_remote_code=True,
        disable_cuda_graph=True,
        log_level="warning",
    )

    processor = AutoProcessor.from_pretrained(model_path)

    def build_inputs(imgs):
        batch = []
        for img in imgs:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch.append({
                "text": text,
                "image_data": img,
                "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
            })
        return batch

    def run_batch(imgs):
        inputs = build_inputs(imgs)
        engine.generate(
            prompt=[inp["text"] for inp in inputs],
            image_data=[inp["image_data"] for inp in inputs],
            sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0},
        )

    def run_all():
        for i in range(0, len(images), batch_size):
            run_batch(images[i: i + batch_size])

    # warm-up: use run_all to trigger all JIT compilations across all images
    print(f"    warm-up ({N_WARMUP}) ...")
    for _ in range(N_WARMUP):
        run_all()

    # timed runs
    times = []
    for r in range(N_RUNS):
        t0 = time.perf_counter()
        run_all()
        elapsed = time.perf_counter() - t0
        ms = elapsed / len(images) * 1000
        times.append(ms)
        print(f"    run {r+1}/{N_RUNS}: {ms:.1f} ms/img")

    engine.shutdown()
    import torch; torch.cuda.empty_cache()
    return times


# ---------------------------------------------------------------------------
# LMDeploy benchmark
# ---------------------------------------------------------------------------
def bench_lmdeploy(images, image_paths, prompt, max_new_tokens, stage, batch_size=1):
    from lmdeploy import pipeline, PytorchEngineConfig
    from lmdeploy.vl import load_image
    from transformers import AutoProcessor

    model_path = STAGE2_MODEL if stage == 2 else STAGE3_MODEL
    tag = f"LMDeploy-bs{batch_size}"
    print(f"\n  [{tag}] loading {Path(model_path).name} ...")

    engine_cfg = PytorchEngineConfig(
        tp=1,
        max_batch_size=batch_size,
        dtype="bfloat16",
        quant_policy=4,        # INT4 weight-only quant
        session_len=512,       # limit session length to save KV cache memory
        cache_max_entry_count=0.15,  # reserve only 15% GPU for KV cache
    )

    pipe = pipeline(model_path, backend_config=engine_cfg)

    def run_batch(imgs):
        # LMDeploy VL: pass (image, text) tuples
        batch = [(img, prompt) for img in imgs]
        pipe(batch, gen_config=dict(max_new_tokens=max_new_tokens, temperature=0.0))

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

    del pipe
    import torch; torch.cuda.empty_cache()
    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["sglang", "lmdeploy", "all"], default="all")
    parser.add_argument("--stage", type=int, choices=[0, 2, 3], default=0)
    args = parser.parse_args()

    engines_to_run = (
        ["sglang", "lmdeploy"] if args.engine == "all" else [args.engine]
    )
    stages_to_run = [2, 3] if args.stage == 0 else [args.stage]

    results = {}

    for stage in stages_to_run:
        prompt   = DISEASE_PROMPT if stage == 2 else CAPTION_PROMPT
        max_tok  = 64 if stage == 2 else 256
        label    = f"Stage {stage}"

        print(f"\n{'='*60}")
        print(f"  {label}  n_samples={N_SAMPLES}  n_runs={N_RUNS}")
        print(f"{'='*60}")

        images, paths = load_images(N_SAMPLES, stage)
        print(f"  Loaded {len(images)} images")

        for engine in engines_to_run:
            for bs in [1, 4]:
                key = f"{engine}-bs{bs}"

                if engine == "sglang":
                    fn = bench_sglang
                elif engine == "lmdeploy":
                    fn = bench_lmdeploy
                else:
                    continue

                try:
                    t = fn(images, paths, prompt, max_tok, stage, batch_size=bs)
                    m, s = stats(t)
                    results[(stage, key)] = (m, s)
                    print(f"  => {key}: {m:.1f} +- {s:.1f} ms/img")
                except Exception as e:
                    print(f"  {key} FAILED: {e}")
                    import traceback; traceback.print_exc()
                    results[(stage, key)] = (None, None)

    # Summary
    all_keys = sorted(set(k for _, k in results.keys()))
    print("\n" + "="*68)
    print(f"{'Backend':<26}  {'Stage 2 ms/img':>20}  {'Stage 3 ms/img':>20}")
    print("-"*68)
    for key in all_keys:
        s2 = results.get((2, key), (None, None))
        s3 = results.get((3, key), (None, None))
        def fmt(r):
            if r is None or r[0] is None:
                return f"{'FAILED':>20}"
            return f"{r[0]:>9.1f} +- {r[1]:>5.1f}"
        print(f"{key:<26}  {fmt(s2)}  {fmt(s3)}")
    print("="*68)

    # Save
    out = {}
    for (stage, key), (m, s) in results.items():
        out.setdefault(f"stage{stage}", {})[key] = {
            "mean_ms_per_img": round(m, 2) if m else None,
            "std_ms_per_img":  round(s, 2) if s else None,
            "imgs_per_sec":    round(1000/m, 2) if m else None,
        }
    out_path = RESULTS_DIR / "results_speed_engines.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
