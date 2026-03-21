"""
eval_bertscore_compare.py
=========================
Compare BERTScore across multiple BERT models on Stage 3 captions.

Models compared:
  1. roberta-large           -- current baseline (general English)
  2. medicalai/ClinicalBERT  -- already used in RAG R1; trained on MIMIC-III clinical notes
  3. ncbi/MedCPT-Article-Encoder -- MedCPT article-side encoder; trained on biomedical articles
  4. microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext -- PubMed abstracts + full text

BERTScore uses the specified model to embed both candidate and reference captions,
then computes token-level cosine similarity (Precision / Recall / F1).
Using a domain-specific BERT gives more accurate similarity for clinical terminology.

Note on MedCPT: The RAG system uses MedCPT-Query-Encoder for short queries and
MedCPT-Article-Encoder for passage retrieval. For comparing two captions,
Article-Encoder is the correct choice.

Run from Windows (CPU) or WSL (GPU):
    python eval_bertscore_compare.py
    python eval_bertscore_compare.py --device cpu   # force CPU
    python eval_bertscore_compare.py --variants merged_noguide merged_guide
"""

import os, json, argparse
from pathlib import Path

os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"]   = "1"

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "disease_classification_results"

PRED_FILES = {
    "merged_noguide": RESULTS_DIR / "results_caption_fuzzytopk_s1cascade_merged_init_stage3_val_predictions.json",
    "merged_guide":   RESULTS_DIR / "results_caption_merged_guided_val_predictions.json",
}

# Models to compare.
# num_layers: which transformer layer to use for embeddings.
#   roberta-large   24 layers -> layer 17 (bert_score default)
#   BERT-base style 12 layers -> layer 9  (bert_score default)
#   Setting None lets bert_score pick the recommended layer automatically.
MODELS = [
    {
        "id":   "roberta-large",
        "label": "RoBERTa-large (general)",
        "hf_name": "roberta-large",
        "num_layers": 17,   # bert_score default for roberta-large (24 layers)
    },
    {
        "id":   "clinical-bert",
        "label": "ClinicalBERT (MIMIC-III clinical notes) — used in RAG R1",
        "hf_name": "medicalai/ClinicalBERT",
        "num_layers": 6,    # ClinicalBERT has only 6 layers; use last layer
    },
    {
        "id":   "medcpt-article",
        "label": "MedCPT-Article-Encoder (PubMed article retrieval) — used in RAG R3",
        "hf_name": "ncbi/MedCPT-Article-Encoder",
        "num_layers": 9,    # BERT-base architecture
    },
    {
        "id":   "pubmedbert",
        "label": "PubMedBERT (PubMed abstracts + full text)",
        "hf_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "num_layers": 9,    # BERT-base architecture
    },
]


def load_predictions(path: Path):
    with open(path) as f:
        d = json.load(f)
    return d["predictions"]


def run_bertscore(predictions, model_info: dict, device: str) -> dict:
    from bert_score import score as bs_score

    cands = [p.get("generated_caption", "") or "" for p in predictions]
    refs  = [p.get("ground_truth_caption", "") or "" for p in predictions]

    kwargs = dict(
        lang="en",
        model_type=model_info["hf_name"],
        num_layers=model_info["num_layers"],
        device=device,
        verbose=False,
        batch_size=16,
    )

    P, R, F1 = bs_score(cands, refs, **kwargs)
    return {
        "P":  round(P.mean().item()  * 100, 2),
        "R":  round(R.mean().item()  * 100, 2),
        "F1": round(F1.mean().item() * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--variants", nargs="+", default=list(PRED_FILES.keys()),
                        help="Which caption variants to evaluate")
    args = parser.parse_args()

    print(f"\nBERTScore model comparison  (device={args.device})")
    print("=" * 80)

    results = {}

    for variant in args.variants:
        path = PRED_FILES.get(variant)
        if path is None or not path.exists():
            print(f"\n[SKIP] {variant}: file not found at {path}")
            continue

        preds = load_predictions(path)
        print(f"\n{'─'*80}")
        print(f"Variant: {variant}  ({len(preds)} predictions)")
        print(f"{'─'*80}")

        results[variant] = {}
        for m in MODELS:
            print(f"\n  [{m['id']}] {m['label']}")
            try:
                scores = run_bertscore(preds, m, args.device)
                results[variant][m["id"]] = scores
                print(f"    P={scores['P']:.2f}  R={scores['R']:.2f}  F1={scores['F1']:.2f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                results[variant][m["id"]] = {"error": str(e)}

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY — BERTScore F1")
    print("=" * 80)
    col_w = 14
    header = f"{'Model':<32}" + "".join(f"{v:>{col_w}}" for v in args.variants)
    print(header)
    print("-" * len(header))
    for m in MODELS:
        row = f"{m['id']:<32}"
        for v in args.variants:
            score = results.get(v, {}).get(m["id"], {})
            val = f"{score['F1']:.2f}" if isinstance(score, dict) and "F1" in score else "ERR"
            row += f"{val:>{col_w}}"
        print(row)
    print("=" * 80)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "results_bertscore_compare.json"
    with open(out_path, "w") as f:
        json.dump({
            "description": "BERTScore comparison across BERT model variants",
            "models": {m["id"]: {"hf_name": m["hf_name"], "label": m["label"]} for m in MODELS},
            "results": results,
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
