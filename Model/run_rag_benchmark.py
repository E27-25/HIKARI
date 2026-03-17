"""
Benchmark runner: iterates over all RAG × Prompt combinations and prints a comparison table.

Usage:
    # Full grid (5 RAG × 4 Prompt = 20 runs)
    python run_rag_benchmark.py

    # Subset
    python run_rag_benchmark.py --rag_experiments R0 R1 R2 --prompt_variants P0 P1

    # Also include No-RAG baseline (USE_RAG=False is handled as "NoRAG" pseudo-experiment)
    python run_rag_benchmark.py --include_no_rag
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Default experiment lists
ALL_RAG_EXPERIMENTS = ["R0", "R1", "R2", "R3", "R4"]
ALL_PROMPT_VARIANTS = ["P0", "P1", "P2", "P3"]

RESULTS_DIR = Path("./disease_classification_results")
INFERENCE_SCRIPT = Path(__file__).parent / "inference_disease_classification.py"


def _result_path(method: str, rag: str, prompt: str, vlm_desc_source: str = "symptoms",
                 alpha: Optional[float] = None) -> Path:
    """Return expected JSON path for a given experiment combo."""
    rag_tag = f"_RAG{rag}" if rag != "NoRAG" else "_NoRAG"
    prompt_tag = f"_P{prompt[1:]}" if prompt != "P0" else ""
    src_tag = f"_s3cap" if vlm_desc_source == "stage3" else ""
    alpha_tag = f"_a{str(alpha).replace('.', '')}" if alpha is not None else ""
    return RESULTS_DIR / f"results_disease_{method}{rag_tag}{prompt_tag}{src_tag}{alpha_tag}_val.json"


def run_experiment(rag: str, prompt: str, batch_size: int, dry_run: bool,
                   n_samples: Optional[int] = None, method: str = "M1",
                   vlm_desc_source: str = "symptoms",
                   alpha: Optional[float] = None) -> bool:
    """Run a single inference experiment. Returns True if successful."""
    if rag == "NoRAG":
        # Patch USE_RAG=False via a temporary env variable override trick
        # Since USE_RAG is a module-level flag, we inject it via environment
        extra_args = ["--rag_exp", "R0"]   # ignored because USE_RAG=False
        env_patch = {"HIKARI_USE_RAG": "0"}
    else:
        extra_args = ["--rag_exp", rag]
        env_patch = {}

    cmd = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--flow", "val",
        "--batch_size", str(batch_size),
        "--prompt", prompt,
        "--stage2_method", method,
        "--vlm_desc_source", vlm_desc_source,
    ] + extra_args

    if alpha is not None:
        cmd += ["--alpha", str(alpha)]
    if n_samples is not None:
        cmd += ["--n_samples", str(n_samples)]

    print(f"\n{'='*60}")
    print(f"Running: RAG={rag}  Prompt={prompt}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("[DRY RUN] Skipped.")
        return True

    import os
    env = os.environ.copy()
    env.update(env_patch)

    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def load_accuracy(result_file: Path) -> Optional[float]:
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("accuracy")
    except Exception:
        return None


def print_table(
    rag_experiments: list,
    prompt_variants: list,
    method: str,
    include_no_rag: bool,
    vlm_desc_source: str = "symptoms",
    alpha: Optional[float] = None,
):
    """Print a RAG × Prompt accuracy table."""
    all_rags = (["NoRAG"] if include_no_rag else []) + rag_experiments

    src_label = f" [vlm={vlm_desc_source}]" if vlm_desc_source != "symptoms" else ""
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS — Method={method}{src_label}")
    print(f"{'='*70}")

    # Header
    col_w = 8
    header = f"{'':10}" + "".join(f"{p:^{col_w}}" for p in prompt_variants)
    print(header)
    print("-" * len(header))

    for rag in all_rags:
        row = f"{rag:10}"
        for prompt in prompt_variants:
            path = _result_path(method, rag, prompt, vlm_desc_source, alpha)
            acc = load_accuracy(path)
            if acc is not None:
                row += f"{acc*100:^{col_w}.2f}"
            else:
                row += f"{'--':^{col_w}}"
        print(row)

    print()


def main():
    parser = argparse.ArgumentParser(description="RAG × Prompt benchmark runner")
    parser.add_argument(
        "--rag_experiments", nargs="+", default=ALL_RAG_EXPERIMENTS,
        metavar="R",
        help="RAG experiment IDs to run (default: R0 R1 R2 R3 R4)",
    )
    parser.add_argument(
        "--prompt_variants", nargs="+", default=ALL_PROMPT_VARIANTS,
        metavar="P",
        help="Prompt variants to run (default: P0 P1 P2 P3)",
    )
    parser.add_argument(
        "--include_no_rag", action="store_true",
        help="Also benchmark with USE_RAG=False (No-RAG baseline row)",
    )
    parser.add_argument(
        "--method", type=str, default="M1",
        help="STAGE2_METHOD used by inference script (default: M1)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip experiments whose result file already exists (default: True)",
    )
    parser.add_argument(
        "--no_skip_existing", dest="skip_existing", action="store_false",
        help="Re-run even if result file already exists",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without running them",
    )
    parser.add_argument(
        "--table_only", action="store_true",
        help="Print summary table from existing results (no new runs)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Limit each run to N samples (smoke test mode)",
    )
    parser.add_argument(
        "--vlm_desc_source", type=str, default="symptoms",
        choices=["symptoms", "stage3", "none"],
        help="Text query source for R1-R3 retrieval: symptoms (default), stage3, none",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Override hybrid score image weight (e.g. 0.7). Default: use index value (0.5).",
    )
    args = parser.parse_args()

    rag_list = args.rag_experiments
    prompt_list = args.prompt_variants
    method = args.method

    if not args.table_only:
        combos = [
            (rag, prompt)
            for rag in rag_list
            for prompt in prompt_list
        ]
        if args.include_no_rag:
            for prompt in prompt_list:
                combos.insert(0, ("NoRAG", prompt))

        print(f"Total experiments: {len(combos)}")
        for rag, prompt in combos:
            out_path = _result_path(method, rag, prompt, args.vlm_desc_source, args.alpha)
            if args.skip_existing and out_path.exists():
                print(f"  [SKIP] {rag}+{prompt} — result exists: {out_path.name}")
                continue
            success = run_experiment(rag, prompt, args.batch_size, args.dry_run, args.n_samples, method=method, vlm_desc_source=args.vlm_desc_source, alpha=args.alpha)
            if not success:
                print(f"  [ERROR] {rag}+{prompt} failed! Continuing...")

    print_table(rag_list, prompt_list, method, args.include_no_rag, vlm_desc_source=args.vlm_desc_source, alpha=args.alpha)


if __name__ == "__main__":
    main()
