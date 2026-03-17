"""
Stage 3 (Caption) Ablation Runner — 4 Experiments

Runs all 4 Stage 3 training + evaluation experiments sequentially:
  Exp 1 (Way 1):       checkpoint init,  no STS  (baseline re-run)
  Exp 2 (Way 2):       merged init,      no STS
  Exp 3 (Way 1 + STS): checkpoint init,  STS+IBR
  Exp 4 (Way 2 + STS): merged init,      STS+IBR

Usage:
  python run_stage3_experiments.py
  python run_stage3_experiments.py --skip 1      # skip Exp 1 (already done)
  python run_stage3_experiments.py --only 3 4    # run Exp 3 and 4 only
"""

import subprocess
import sys
import json
import time
import argparse
from pathlib import Path

# Fix Windows console encoding — replace unencodable chars instead of crashing
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass  # Python < 3.7


# ============================================================================
# EXPERIMENT TABLE
# ============================================================================

EXPERIMENTS = [
    {
        "id": 1,
        "name": "Way 1 (checkpoint init, no STS)",
        "train_flags": ["--start_from_stage1", "--stage3_init", "checkpoint"],
        "eval_method": "fuzzytopk_s1cascade_stage3",
        "output_merged": "./skincap_stage3_caption_fuzzytopk_s1cascade_classification_merged",
        "log_file": "stage3_exp1_way1.log",
    },
    {
        "id": 2,
        "name": "Way 2 (merged init, no STS)",
        "train_flags": ["--start_from_stage1", "--stage3_init", "merged"],
        "eval_method": "fuzzytopk_s1cascade_merged_init_stage3",
        "output_merged": "./skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_classification_merged",
        "log_file": "stage3_exp2_way2.log",
    },
    {
        "id": 3,
        "name": "Way 1 + STS (checkpoint init, STS+IBR)",
        "train_flags": ["--start_from_stage1", "--stage3_init", "checkpoint", "--use_sts"],
        "eval_method": "fuzzytopk_s1cascade_sts_stage3",
        "output_merged": "./skincap_stage3_caption_fuzzytopk_s1cascade_sts_classification_merged",
        "log_file": "stage3_exp3_way1_sts.log",
    },
    {
        "id": 4,
        "name": "Way 2 + STS (merged init, STS+IBR)",
        "train_flags": ["--start_from_stage1", "--stage3_init", "merged", "--use_sts"],
        "eval_method": "fuzzytopk_s1cascade_merged_init_sts_stage3",
        "output_merged": "./skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_sts_classification_merged",
        "log_file": "stage3_exp4_way2_sts.log",
    },
]

TRAIN_SCRIPT = Path(__file__).parent / "train_two_stage_FuzzyTopK.py"
EVAL_SCRIPT  = Path(__file__).parent / "inference_disease_classification.py"
RESULTS_FILE = Path(__file__).parent / "stage3_ablation_results.json"


# ============================================================================
# HELPERS
# ============================================================================

def run_cmd(cmd, log_path: Path, label: str) -> bool:
    """Run a subprocess, tee stdout+stderr to log file, return success."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    print(f"  LOG: {log_path}")
    print(f"{'='*60}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with open(log_path, "w", encoding="utf-8") as flog:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            try:
                print(line, end="")
            except UnicodeEncodeError:
                print(line.encode("ascii", errors="replace").decode("ascii"), end="")
            flog.write(line)
        proc.wait()

    elapsed = time.time() - t0
    mins = int(elapsed // 60)
    print(f"\n  [{'OK' if proc.returncode == 0 else 'FAIL'}] Elapsed: {mins}m {int(elapsed%60)}s")
    return proc.returncode == 0


def load_caption_results(method: str) -> dict:
    """Load BLEU/ROUGE results from inference output JSON."""
    results_dir = Path(__file__).parent / "disease_classification_results"
    # Try both val and whole split filenames
    for split in ("val", "whole"):
        candidate = results_dir / f"results_caption_{method}_{split}.json"
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                return json.load(f)
    # Fallback: look for any file with method in name
    for p in sorted(results_dir.glob(f"*{method}*")):
        if p.suffix == ".json" and "caption" in p.name:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return {}


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 3 ablation runner")
    parser.add_argument("--skip", type=int, nargs="+", default=[],
                        help="Experiment IDs to skip (e.g. --skip 1)")
    parser.add_argument("--only", type=int, nargs="+", default=[],
                        help="Run only these experiment IDs (e.g. --only 3 4)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, run evaluation only")
    args = parser.parse_args()

    # Filter experiments
    exps = EXPERIMENTS
    if args.only:
        exps = [e for e in exps if e["id"] in args.only]
    elif args.skip:
        exps = [e for e in exps if e["id"] not in args.skip]

    print(f"\nStage 3 Ablation — running {len(exps)} experiment(s)")
    for e in exps:
        print(f"  Exp {e['id']}: {e['name']}")

    all_results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            all_results = json.load(f)

    # ------------------------------------------------------------------ #
    # RUN EACH EXPERIMENT                                                  #
    # ------------------------------------------------------------------ #
    for exp in exps:
        exp_id = exp["id"]
        name   = exp["name"]
        log    = Path(__file__).parent / exp["log_file"]

        print(f"\n{'#'*60}")
        print(f"# Experiment {exp_id}: {name}")
        print(f"{'#'*60}")

        # ---- TRAINING ------------------------------------------------- #
        if not args.eval_only:
            merged_path = Path(__file__).parent / exp["output_merged"]
            if merged_path.exists():
                print(f"[SKIP TRAIN] Merged model already exists: {merged_path}")
            else:
                train_cmd = [
                    sys.executable, str(TRAIN_SCRIPT),
                    "--mode", "stage2",
                ] + exp["train_flags"]

                ok = run_cmd(train_cmd, log, f"TRAINING Exp {exp_id}: {name}")
                if not ok:
                    print(f"[ERROR] Training failed for Exp {exp_id} — skipping eval")
                    all_results[str(exp_id)] = {"error": "training_failed", "name": name}
                    continue
        else:
            print(f"[eval_only] Skipping training for Exp {exp_id}")

        # ---- EVALUATION ----------------------------------------------- #
        method = exp["eval_method"]
        eval_log = log.with_name(log.stem + "_eval.log")

        eval_cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--flow", "val",
            "--stage2_method", method,
            "--batch_size", "4",
        ]

        ok = run_cmd(eval_cmd, eval_log, f"EVAL Exp {exp_id}: {name}")

        # Load BLEU/ROUGE from JSON output
        caption_results = load_caption_results(method)
        if caption_results:
            bleu4 = caption_results.get("bleu_4", "N/A")
            rouge_l = caption_results.get("rouge_l", "N/A")
            print(f"\n  *** Exp {exp_id} Results: BLEU-4={bleu4}  ROUGE-L={rouge_l} ***")
            all_results[str(exp_id)] = {
                "name": name,
                "method": method,
                "bleu_1": caption_results.get("bleu_1"),
                "bleu_2": caption_results.get("bleu_2"),
                "bleu_3": caption_results.get("bleu_3"),
                "bleu_4": caption_results.get("bleu_4"),
                "rouge_1": caption_results.get("rouge_1"),
                "rouge_2": caption_results.get("rouge_2"),
                "rouge_l": caption_results.get("rouge_l"),
            }
        else:
            print(f"  [WARN] Could not find eval JSON for method={method}")
            all_results[str(exp_id)] = {"name": name, "method": method, "error": "no_eval_json"}

        # Save intermediate results after each experiment
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to: {RESULTS_FILE}")

    # ------------------------------------------------------------------ #
    # FINAL SUMMARY TABLE                                                  #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("  STAGE 3 ABLATION — FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Exp':<5} {'Name':<38} {'BLEU-4':>8} {'ROUGE-L':>8}")
    print(f"  {'-'*5} {'-'*38} {'-'*8} {'-'*8}")

    # Baseline (existing result)
    print(f"  {'0':<5} {'Baseline (original 9.82%)':<38} {'9.82':>8} {'':>8}")

    for exp in EXPERIMENTS:
        r = all_results.get(str(exp["id"]), {})
        b4 = f"{r.get('bleu_4', '?'):>8}" if isinstance(r.get("bleu_4"), (int, float)) else f"{'?':>8}"
        rl = f"{r.get('rouge_l', '?'):>8}" if isinstance(r.get("rouge_l"), (int, float)) else f"{'?':>8}"
        print(f"  {exp['id']:<5} {exp['name']:<38} {b4} {rl}")

    print(f"{'='*70}")
    print(f"\n  Full results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
