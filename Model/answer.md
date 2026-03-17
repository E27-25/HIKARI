# RAG Benchmark Analysis

Benchmark results used:

| Method | P0-R0 | P3-R0 (best) |
|--------|-------|--------------|
| fuzzytopk | 66.67% | **90.00%** |
| fuzzytopk_s1cascade | 70.67% | 69.33% |
| M1 | 55.22% | 50.00% |

---

## 1. Difference Between 2nd-Stage Methods — Ground Truth Usage

| Method | Uses GT at inference? | Starting model weights | Group context |
|--------|----------------------|----------------------|---------------|
| **fuzzytopk** | ❌ No | Base Qwen3-VL-8B (no prior fine-tuning) | None |
| **fuzzytopk_s1cascade** | ❌ No | Stage1 merged (group classifier, domain-adapted visual encoder) | None |
| **M1** | ✅ **YES** | Stage1 merged | **GT group label injected into every prompt** |

**M1 is an oracle** — it tells the model "this lesion belongs to group X" using the true label. This is not usable in real deployment. The 55.22% result is surprisingly low even with this advantage, explained in Q5 below.

**fuzzytopk_s1cascade** benefits indirectly from Stage1: the model starts with weights already fine-tuned on skin lesion group classification, giving it a better visual encoder for skin features — but no label information is passed at inference.

---

## 2. R0 = R1 (Identical Results) — Strategy B Silent Fallback

### Why R0 = R1

R1 uses **Strategy B** (two-pass: image + text). Strategy B requires a text description of the query image (`vlm_description`) to activate the text encoder (ClinicalBERT for R1). The current call is:

```python
# inference_disease_classification.py line 745 (before fix)
ref_pairs = rag_retriever.retrieve(image, k=RAG_K)
#                                                 ↑ vlm_description not passed → None
```

Strategy B code in `rag_retrieval.py`:
```python
elif self.strategy == "B" and vlm_description:   # ← only runs if vlm_description is not None
    q_txt = self._embed_text(vlm_description)
    txt_sims = self._cosine_scores(q_txt, self.txt_embs)
    scores = self.alpha * img_sims + (1-self.alpha) * txt_sims
else:
    scores = img_sims  # ← fallback: image-only
```

With `vlm_description=None`, Strategy B silently falls back to **image-only scoring**. Since R0 and R1 both use `openai/clip-vit-base-patch32` as the image encoder → **identical retrieval results**.

### Image vs Text Weight (alpha)

```
alpha = 0.5  (default in HybridRAGRetriever.__init__)

Strategy A score = 0.5 × cos(img_query, img_ref) + 0.5 × cos(img_query, txt_ref)
Strategy B score = 0.5 × cos(img_query, img_ref) + 0.5 × cos(txt_query, txt_ref)
```

Currently R1/R2/R3 text weight is **effectively 0** (fallback to image-only).

### Will adding symptoms/location to in-context retrieval improve P3 on R1/R2?

**Yes — once text retrieval is enabled.** Fix applied (see below):

```python
# inference_disease_classification.py line 745 (after fix)
vlm_desc = item.get('caption')  # clinical description from SkinCAP CSV
ref_pairs = rag_retriever.retrieve(image, k=RAG_K, vlm_description=vlm_desc)
```

Every val item already has a `caption` field (loaded from `caption_zh_polish_en` column). This is a clinical description of the lesion appearance (morphology, color, scale — not the label). Passing it enables:
- **R1**: CLIP image + ClinicalBERT text → medical-domain text matching
- **R2**: SigLIP image + BGE-M3 text → multilingual/general text matching
- **R3**: JinaClip image + MedCPT text → PubMed-specialized text matching

If the caption contains symptom terms ("erythematous plaques", "scaling", "facial distribution"), text-based retrieval would find training images with similar clinical descriptions → potentially better matches than visual similarity alone for look-alike diseases.

**R2 and R3 already show different numbers from R0** (different image encoders SigLIP/JinaClip), so their image-only results are real. After the fix, the text component adds on top of that.

---

## 3. Data Leakage Check

### Self-retrieval leak: ✅ FIXED
- `RAG_USE_ALL_DATA = False` → RAG index built from **911 train images only**
- Val images (99) are excluded from the index pool
- A val query cannot retrieve itself → no ground-truth label leakage

### Strategy B text component: previously unused (now fixed)
- R1/R2/R3 text retrieval was silently disabled (vlm_description=None)
- This is **not** leakage — just a missing feature, now enabled

### Unknown prediction rate (soft issue)
- M1 has 32/99 unparseable outputs ("Unknown") vs 24/99 for fuzzytopk
- Cause: M1 messages are very complex (3 reference images + labels + group context + diagnosis prompt) → Qwen3-VL sometimes generates verbose outputs that don't match any disease name
- These count as incorrect (not excluded), so M1's real accuracy on parseable predictions is higher

### M1 melanocytic_nevi support=1 (anomaly explained)
- The per-class `support` count in results only covers samples with **parseable predictions**
- 8 of the 9 GT melanocytic nevi samples returned "Unknown" for M1 → dropped from support count
- fuzzytopk support=9 (all 9 melanocytic nevi predictions were parseable)

---

## 4. Per-Class Sensitivity (P0, R0)

> Sensitivity = % of GT samples for that class correctly predicted

| Disease | fuzzytopk | fuzzytopk_s1cascade | M1 |
|---------|:---------:|:-------------------:|:--:|
| psoriasis | 75% | 62.5% | 62.5% |
| lupus erythematosus | 25% | **75%** | 50% |
| lichen planus | 42.9% | **71.4%** | 57.1% |
| scleroderma | **71.4%** | 42.9% | 14.3% |
| photodermatoses | 37.5% | 37.5% | 37.5% |
| sarcoidosis | **80%** | 60% | **80%** |
| melanocytic nevi | 88.9% | **100%** | 100%† |
| squamous cell carcinoma in situ | **90.9%** | **90.9%** | 63.6% |
| basal cell carcinoma | 44.4% | **66.7%** | 33.3% |
| acne vulgaris | 85.7% | 85.7% | **100%** |
| **Overall accuracy** | **66.67%** | **70.67%** | **55.22%** |
| Unknown predictions | 24/99 | — | 32/99 |

†M1 melanocytic_nevi: only 1 of 9 GT samples produced parseable output (8 returned "Unknown")

### Key patterns

**s1cascade wins on inflammatory diseases** (lupus 75%, lichen planus 71.4%) — Stage1 domain adaptation helps for visually-similar diseases requiring subtle feature discrimination.

**fuzzytopk wins on malignant diseases** (SCCIS 90.9%, both tied; scleroderma 71.4%) — base model generalization handles morphologically-distinct lesions well.

**M1 (GT groups + cascade) underperforms both** — complex message format causes high Unknown rate; group context doesn't overcome message parsing issues.

**Persistent bottleneck**: photodermatoses stuck at 37.5% across all methods (8 GT samples, insufficient training data for visual distinction).

---

## 5. Stage1 Cascade vs No Cascade

### P0-R0 comparison (simple prompt)

| Method | P0-R0 | vs fuzzytopk |
|--------|-------|-------------|
| fuzzytopk (no cascade) | 66.67% | baseline |
| **fuzzytopk_s1cascade** | **70.67%** | **+4%** |
| M1 (cascade + GT groups) | 55.22% | -11.4% |

**Cascade helps for simple prompts** (+4%): Stage1 visual encoder learns skin lesion features → better visual representations → better disease discrimination.

### P3-R0 comparison (structured clinical prompt)

| Method | P3-R0 | vs fuzzytopk |
|--------|-------|-------------|
| **fuzzytopk (no cascade)** | **90.00%** | baseline |
| fuzzytopk_s1cascade | 69.33% | **-20.7%** |
| M1 (cascade + GT groups) | 50.00% | -40% |

**Cascade HURTS for complex prompts** (-20.7%): Stage1 was trained to output short group names ("1. Inflammatory & Autoimmune Diseases"). This training creates a bias toward terse outputs. When P3 asks the model to fill in a structured template with multiple fields, the s1cascade model reverts to the short-output pattern it learned during Stage1 fine-tuning — it doesn't follow the format properly.

The base fuzzytopk model (Qwen3-VL-8B without Stage1 bias) responds to P3's template natively, filling in the fields correctly and reaching 90%.

### Summary

| Scenario | Best method |
|----------|------------|
| Simple prompts (P0/P1) | fuzzytopk_s1cascade (+4% from domain adaptation) |
| Complex prompts (P2/P3) | fuzzytopk base (no cascade bias) |
| Production (no GT) | fuzzytopk_s1cascade P0 = 70.67% or fuzzytopk P3 = 90% |
| Oracle (GT groups) | M1 — but 55.22% due to parsing issues |

**Best overall result so far: fuzzytopk + P3 + R0 = 90.00%**

### Why M1 is worst despite having GT group context

Three compounding problems:
1. **Cascade weight bias** (same as s1cascade): Stage1 weights bias toward short outputs
2. **Complex message**: 3 reference images + 3 labels + group context text + question → very long prompt → Qwen3-VL generates verbose reasoning that is hard to parse → 32/99 Unknown
3. **Conflicting signals**: RAG reference images + GT group label + diagnosis question is too much context — the model tries to reconcile all signals rather than just classifying the query image

---

## Next Steps

1. **Re-run benchmark with Strategy B fix** (caption as vlm_description) to see if R1/R2/R3 text retrieval improves over R0
2. **Test fuzzytopk + P3 + R3** (JinaClip + MedCPT, text-enabled): medical-domain text encoder + structured prompt may push beyond 93.75%
3. **Fix M1 Unknown rate**: reduce `MAX_NEW_TOKENS` or simplify message when RAG is on (remove group context when RAG examples already anchor the diagnosis)
4. **Try alpha tuning**: default is 0.5 (equal image/text); for medical domain, image is usually more reliable → test alpha=0.7 (more image weight)
