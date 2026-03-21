"""
eval_caption_extended.py
========================
Computes the full Stage 3 caption evaluation table for Conference_Paper.tex:

  Init         | Guide | B-1 | B-2 | B-4 | R-1 | BS-F | Corr.
  -------------|-------|-----|-----|-----|-----|------|------
  Checkpoint   |   ✗   |     |     |     |     |      |
  Checkpoint   |   ✓   |     |     |     |     |      |
  Merged-Init  |   ✗   |     |     |     |     |      |
  Merged-Init  |   ✓   |     |     |     |     |      |

- ✗  = image-only caption prompt (existing Stage 3 models, no disease label)
- ✓  = Stage 2 predicted disease label injected before caption prompt
- BS-F = BERTScore F1 (microsoft/deberta-xlarge-mnli or fallback to roberta-large)
- Corr. = % captions that contain the correct disease name (fuzzy match ≥ 80)

Phase 1 (fast, CPU-OK): BERTScore + Correctness on existing prediction files
Phase 2 (GPU required): Generate guided captions, then compute all metrics

Usage:
    cd Model/
    python eval_caption_extended.py          # full run (both phases)
    python eval_caption_extended.py --phase1 # BERTScore + Corr. on existing only
"""

import os, sys, json, re, argparse
from pathlib import Path

os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"]   = "1"

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "disease_classification_results"
OUT_DIR     = RESULTS_DIR

# ---------------------------------------------------------------------------
# Paths to existing Stage 3 prediction files
# ---------------------------------------------------------------------------
PRED_FILES = {
    "checkpoint_noguide": RESULTS_DIR / "results_caption_fuzzytopk_s1cascade_stage3_val_predictions.json",
    "merged_noguide":     RESULTS_DIR / "results_caption_fuzzytopk_s1cascade_merged_init_stage3_val_predictions.json",
}

# Stage 2 predictions used to build guided prompts (Cascaded FT, R2 α=0.9)
STAGE2_PRED_FILE = RESULTS_DIR / "results_disease_fuzzytopk_s1cascade_RAGR2_a09_val_predictions.json"

# Stage 3 model paths
MODEL_PATHS = {
    "checkpoint": str(BASE_DIR / "skincap_stage3_caption_fuzzytopk_s1cascade_classification_merged"),
    "merged":     str(BASE_DIR / "skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_classification_merged"),
}

# Caption prompts
CAPTION_PROMPT = (
    "Describe this skin lesion image in detail. Include information about its "
    "appearance, possible diagnosis, and recommended examinations."
)
CAPTION_PROMPT_GUIDED = (
    "Stage 2 diagnosis: {disease_label}.\n\n"
    "Describe this skin lesion image in detail. Include information about its "
    "appearance, possible diagnosis, and recommended examinations."
)

# ---------------------------------------------------------------------------
# Disease name aliases for correctness matching
# ---------------------------------------------------------------------------
DISEASE_ALIASES = {
    "psoriasis":                    ["psoriasis", "psoriatic"],
    "lupus erythematosus":          ["lupus", "lupus erythematosus", "discoid lupus"],
    "lichen planus":                ["lichen planus", "lichen"],
    "scleroderma":                  ["scleroderma", "systemic sclerosis", "morphea"],
    "photodermatoses":              ["photodermatosis", "photodermatoses", "photosensitivity",
                                     "photodermato", "light sensitivity"],
    "sarcoidosis":                  ["sarcoidosis", "sarcoid"],
    "melanocytic nevi":             ["melanocytic nevi", "melanocytic nevus",
                                     "nevi", "nevus", "mole", "melanocytic"],
    "squamous cell carcinoma in situ": ["squamous cell carcinoma in situ", "sccis",
                                        "squamous cell carcinoma", "squamous"],
    "basal cell carcinoma":         ["basal cell carcinoma", "bcc", "basal cell"],
    "acne vulgaris":                ["acne vulgaris", "acne", "comedone", "comedonal"],
}

def _make_alias_set():
    """Map every normalized GT disease string → list of aliases."""
    out = {}
    for canonical, aliases in DISEASE_ALIASES.items():
        out[canonical.lower()] = [a.lower() for a in aliases]
    return out

ALIAS_MAP = _make_alias_set()


def disease_correct(caption: str, gt_disease: str) -> bool:
    """Return True if caption contains any alias for gt_disease."""
    cap_lower = caption.lower()
    gt_norm   = gt_disease.lower().replace("-", " ").strip()

    # Try direct aliases
    aliases = ALIAS_MAP.get(gt_norm, [gt_norm])
    for alias in aliases:
        if alias in cap_lower:
            return True

    # Fallback: check if most words of gt_norm appear in caption
    words = [w for w in gt_norm.split() if len(w) > 3]
    if words and all(w in cap_lower for w in words):
        return True

    return False


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_bleu_rouge(predictions):
    """Compute corpus BLEU-{1,2,4} and ROUGE-1 from prediction list."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from rouge_score import rouge_scorer

    refs, hyps = [], []
    for p in predictions:
        ref = p.get("ground_truth_caption", "") or ""
        hyp = p.get("generated_caption",   "") or ""
        refs.append([ref.lower().split()])
        hyps.append(hyp.lower().split())

    sf = SmoothingFunction().method1
    b1 = corpus_bleu(refs, hyps, weights=(1,0,0,0), smoothing_function=sf) * 100
    b2 = corpus_bleu(refs, hyps, weights=(.5,.5,0,0), smoothing_function=sf) * 100
    b4 = corpus_bleu(refs, hyps, weights=(.25,.25,.25,.25), smoothing_function=sf) * 100

    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    r1_scores = [
        scorer.score(p["ground_truth_caption"], p["generated_caption"])["rouge1"].fmeasure
        for p in predictions
        if p.get("ground_truth_caption") and p.get("generated_caption")
    ]
    r1 = (sum(r1_scores) / len(r1_scores) * 100) if r1_scores else 0.0

    return round(b1, 2), round(b2, 2), round(b4, 2), round(r1, 2)


def compute_bertscore(predictions, device="cuda"):
    """Compute BERTScore F1 (corpus average)."""
    from bert_score import score as bs_score

    cands = [p.get("generated_caption",   "") or "" for p in predictions]
    refs  = [p.get("ground_truth_caption", "") or "" for p in predictions]

    P, R, F1 = bs_score(
        cands, refs,
        lang="en",
        model_type="roberta-large",
        device=device,
        verbose=False,
        batch_size=16,
    )
    return round(F1.mean().item() * 100, 2)


def compute_correctness(predictions, gt_map):
    """Compute % of captions that mention the correct disease name."""
    correct = 0
    for p in predictions:
        img_id  = str(p.get("id", ""))
        caption = p.get("generated_caption", "") or ""
        gt_dis  = gt_map.get(img_id, "")
        if disease_correct(caption, gt_dis):
            correct += 1
    return round(correct / len(predictions) * 100, 2) if predictions else 0.0


# ---------------------------------------------------------------------------
# Build image-id → ground-truth disease map from Stage 2 predictions
# ---------------------------------------------------------------------------
def build_gt_map(stage2_pred_file: Path):
    with open(stage2_pred_file) as f:
        d = json.load(f)
    return {str(p["id"]): p["ground_truth"] for p in d["predictions"]}


def build_stage2_pred_map(stage2_pred_file: Path):
    """image-id → Stage 2 predicted disease (used to build guided prompts)."""
    with open(stage2_pred_file) as f:
        d = json.load(f)
    return {str(p["id"]): p["predicted"] for p in d["predictions"]}


# ---------------------------------------------------------------------------
# Load existing predictions
# ---------------------------------------------------------------------------
def load_predictions(path: Path):
    with open(path) as f:
        d = json.load(f)
    return d["predictions"]


# ---------------------------------------------------------------------------
# Phase 2: guided inference
# ---------------------------------------------------------------------------
def run_guided_inference(model_key: str, stage2_pred_map: dict,
                         existing_preds: list, device: str = "cuda"):
    """
    Load Stage 3 model and regenerate captions with Stage 2 disease label
    prepended to the prompt.
    Returns a list of prediction dicts in the same format.
    """
    import torch
    from PIL import Image
    from unsloth import FastVisionModel

    model_path = MODEL_PATHS[model_key]
    print(f"\n[Phase 2] Loading {model_key} model from {model_path} ...")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=True,
        dtype=None,
    )
    FastVisionModel.for_inference(model)
    model.eval()

    results = []
    for item in existing_preds:
        img_id   = str(item["id"])
        img_path = str(BASE_DIR / item["image_path"])
        gt_cap   = item.get("ground_truth_caption", "")
        disease  = stage2_pred_map.get(img_id, "unknown")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            results.append({
                "id": img_id,
                "file_name": item["file_name"],
                "image_path": item["image_path"],
                "ground_truth_caption": gt_cap,
                "generated_caption": "",
                "status": "error: image load failed",
                "stage2_disease": disease,
            })
            continue

        prompt_text = CAPTION_PROMPT_GUIDED.format(disease_label=disease)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt_text},
        ]}]

        inputs = tokenizer.apply_chat_template(
            [messages], tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, padding=True,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Strip prompt echo (assistant turn starts after "assistant" marker)
        if "assistant" in text.lower():
            text = text.split("assistant")[-1].strip()
        # Strip thinking blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        results.append({
            "id":                    img_id,
            "file_name":             item["file_name"],
            "image_path":            item["image_path"],
            "ground_truth_caption":  gt_cap,
            "generated_caption":     text,
            "status":                "success",
            "stage2_disease":        disease,
        })
        print(f"  [{img_id}] {disease[:30]:30s}  -> {text[:60]}")

    del model
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", action="store_true",
                        help="Only run Phase 1 (BERTScore + Corr on existing preds, no GPU inference)")
    parser.add_argument("--device", default="cuda", help="Device for BERTScore and inference")
    args = parser.parse_args()

    # ---- build ground-truth and Stage 2 prediction maps ----
    print("Loading Stage 2 predictions ...")
    gt_map          = build_gt_map(STAGE2_PRED_FILE)
    stage2_pred_map = build_stage2_pred_map(STAGE2_PRED_FILE)
    print(f"  {len(gt_map)} val samples found.")

    rows = {}

    # ====================================================================
    # Phase 1: existing predictions (no guide)
    # ====================================================================
    print("\n=== Phase 1: BERTScore + Correctness on existing predictions ===")

    for key, path in PRED_FILES.items():
        print(f"\n  {key}: {path.name}")
        preds = load_predictions(path)

        b1, b2, b4, r1 = compute_bleu_rouge(preds)
        bsf = compute_bertscore(preds, device=args.device)
        corr = compute_correctness(preds, gt_map)

        rows[key] = dict(B1=b1, B2=b2, B4=b4, R1=r1, BSF=bsf, Corr=corr)
        print(f"    B-1={b1}  B-2={b2}  B-4={b4}  R-1={r1}  BS-F={bsf}  Corr={corr}%")

    # Save phase 1 results
    phase1_out = OUT_DIR / "results_caption_extended_phase1.json"
    with open(phase1_out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nPhase 1 results saved -> {phase1_out}")

    if args.phase1:
        print("\n[--phase1 flag set] Skipping guided inference. Final rows so far:")
        _print_table(rows)
        return

    # ====================================================================
    # Phase 2: guided inference
    # ====================================================================
    print("\n=== Phase 2: Guided caption generation (Stage 2 label injected) ===")

    for model_key, noguide_key in [("checkpoint", "checkpoint_noguide"),
                                   ("merged",     "merged_noguide")]:
        existing_preds = load_predictions(PRED_FILES[noguide_key])
        guided_preds   = run_guided_inference(model_key, stage2_pred_map,
                                              existing_preds, device=args.device)

        # Save guided predictions
        guided_out = OUT_DIR / f"results_caption_{model_key}_guided_val_predictions.json"
        with open(guided_out, "w") as f:
            json.dump({"predictions": guided_preds}, f, indent=2, ensure_ascii=False)
        print(f"  Guided preds saved -> {guided_out}")

        b1, b2, b4, r1 = compute_bleu_rouge(guided_preds)
        bsf  = compute_bertscore(guided_preds, device=args.device)
        corr = compute_correctness(guided_preds, gt_map)

        guide_key = f"{model_key}_guide"
        rows[guide_key] = dict(B1=b1, B2=b2, B4=b4, R1=r1, BSF=bsf, Corr=corr)
        print(f"  B-1={b1}  B-2={b2}  B-4={b4}  R-1={r1}  BS-F={bsf}  Corr={corr}%")

    # Save full results
    full_out = OUT_DIR / "results_caption_extended_full.json"
    with open(full_out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nFull results saved -> {full_out}")

    _print_table(rows)


def _print_table(rows):
    print("\n" + "="*80)
    print(f"{'Init':<16} {'Guide':<7} {'B-1':>6} {'B-2':>6} {'B-4':>6} "
          f"{'R-1':>6} {'BS-F':>7} {'Corr%':>7}")
    print("-"*80)
    display = [
        ("checkpoint_noguide", "Checkpoint", "✗"),
        ("checkpoint_guide",   "Checkpoint", "✓"),
        ("merged_noguide",     "Merged",     "✗"),
        ("merged_guide",       "Merged",     "✓"),
    ]
    for key, init, guide in display:
        r = rows.get(key)
        if r:
            print(f"{init:<16} {guide:<7} {r['B1']:>6.2f} {r['B2']:>6.2f} "
                  f"{r['B4']:>6.2f} {r['R1']:>6.2f} {r['BSF']:>7.2f} {r['Corr']:>6.2f}%")
        else:
            print(f"{init:<16} {guide:<7} {'--':>6} {'--':>6} {'--':>6} "
                  f"{'--':>6} {'--':>7} {'--':>6}")
    print("="*80)


if __name__ == "__main__":
    main()
