"""
Attention Map Visualization: HIKARI vs Base Qwen3-VL-8B
=========================================================
Extracts language-model attention from the FIRST GENERATED TOKEN back to
every image-patch position in the input sequence.  This directly answers
"which image regions did the model attend to when predicting the disease?"

Method:
  1. Load model with attn_implementation="eager" (materialises attention weights).
  2. Find <|image_pad|> token positions in input_ids.
  3. Run generate(max_new_tokens=1, output_attentions=True).
  4. From out.attentions[0] (first generated token), average the attention
     weights over all transformer layers and all heads.
  5. Slice to image-pad positions → reshape → upsample to IMG_SIZE×IMG_SIZE.

Saves to: gradcam_outputs/
  {disease}_comparison.png   -- side-by-side (original | base | HIKARI)
  {disease}_base.png         -- base model overlay only
  {disease}_hikari.png       -- HIKARI overlay only

Usage:
    python gradcam_visualization.py
"""

import csv, gc, json, math, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image
from scipy.ndimage import zoom

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_BASE   = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_HIKARI = "./skincap_fuzzytopk_classification_merged"
SPLIT_INFO   = "./split_info_3stage.json"
CSV_PATH     = "./SkinCAP/skincap_v240623.csv"
IMG_DIR      = "./SkinCAP/skincap"
OUT_DIR      = Path("gradcam_outputs")
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE   = 336  # input image size (pixels)
N_SAMPLES  = 5    # candidates per disease to try

TARGETS = [
    "psoriasis",            # +38.5pp improvement (61.5% → 100%) — biggest winner
    "basal cell carcinoma", # +23.1pp improvement (76.9% → 100%) — clear visible lesion
    "sarcoidosis",          # -42.8pp collapse (57.1% → 14.3%)  — failure case (paper limitation §V-B)
    "melanocytic nevi",     # +8.3pp (91.7% → 100%)            — benign control
]

PROMPT = ("Examine this skin lesion carefully. "
          "What specific skin disease is shown? "
          "Focus on the lesion morphology, color, and surface texture.")

# ── Image loading ──────────────────────────────────────────────────────────────

def load_img(path: str):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return img, np.array(img)

# ── Attention extraction ───────────────────────────────────────────────────────

def _build_messages(img_pil):
    return [{"role": "user", "content": [
        {"type": "image", "image": img_pil},
        {"type": "text",  "text": PROMPT},
    ]}]


def extract_attention(model_path: str, img_pil, device="cuda", label="model"):
    """
    Extract WHERE the model focuses when predicting the disease.

    Uses language-model attention: the first generated token attends back
    to the full input sequence including all image-pad positions.  That
    attention (averaged over all layers and heads, sliced to image positions)
    is the ground-truth signal for which patches drove the prediction.

    Requires attn_implementation="eager" so that attention weight tensors
    are materialised (Flash-Attention / SDPA do not expose them).
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from transformers import BitsAndBytesConfig

    print(f"  [{label}] Loading ...")
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    bnb  = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_compute_dtype=torch.float16)
    mdl  = Qwen3VLForConditionalGeneration.from_pretrained(
               model_path,
               quantization_config=bnb,
               device_map=device,
               trust_remote_code=True,
               attn_implementation="eager",   # must materialise attn weights
           )
    mdl.eval()

    # ── Build inputs ──────────────────────────────────────────────────────
    msgs   = _build_messages(img_pil)
    text   = proc.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=True)
    inputs = proc(text=[text], images=[img_pil],
                  return_tensors="pt").to(device)

    # ── Locate <|image_pad|> positions in input_ids ───────────────────────
    input_ids  = inputs["input_ids"][0].cpu()
    img_tok_id = getattr(mdl.config, "image_token_id", None)
    if img_tok_id is None:
        img_tok_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    img_pos = (input_ids == img_tok_id).nonzero(as_tuple=True)[0]  # (n_img,)
    n_img   = len(img_pos)

    if n_img == 0:
        print(f"  [{label}] WARNING: 0 image tokens found — check tokenizer")
        del mdl; torch.cuda.empty_cache(); gc.collect()
        return None
    print(f"  [{label}] {n_img} image tokens "
          f"(pos {img_pos[0].item()}–{img_pos[-1].item()})")

    # ── Prefill pass: capture attention from the LAST INPUT TOKEN → image ────
    # The last input token is the generation prompt end ("just before answering,
    # what does the model attend to in the image?"). We use only the last 4
    # transformer layers — they carry the most task-specific signal.
    # Full attention matrices are (batch, heads, seq_len, seq_len) per layer;
    # storing all 28 layers simultaneously is ~320 MB — fine on 16 GB VRAM.
    with torch.no_grad():
        out = mdl(**inputs, output_attentions=True)

    if not getattr(out, "attentions", None):
        print(f"  [{label}] WARNING: output_attentions unsupported")
        del mdl, out; torch.cuda.empty_cache(); gc.collect()
        return None

    seq_len    = inputs["input_ids"].shape[1]
    last_pos   = seq_len - 1          # index of last input token
    layer_maps = []

    for la in out.attentions[-4:]:    # LAST 4 LAYERS only (most task-specific)
        if la is None:
            continue
        # la: (1, heads, seq_len, seq_len)
        # row = last_pos → attention FROM last token TO each key position
        la_last = la[0, :, last_pos, :].float().cpu()   # (heads, seq_len)
        la_img  = la_last[:, img_pos]                   # (heads, n_img)
        layer_maps.append(la_img.mean(0))               # mean over heads → (n_img,)

    if not layer_maps:
        del mdl, out; torch.cuda.empty_cache(); gc.collect()
        return None

    img_attn = torch.stack(layer_maps).mean(0).numpy()  # avg over 4 layers

    # ── Reshape to spatial grid and upsample ─────────────────────────────
    side     = round(math.sqrt(n_img))
    img_attn = img_attn[:side * side]
    heat     = img_attn.reshape(side, side)
    heat     = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat     = zoom(heat, (IMG_SIZE / side, IMG_SIZE / side), order=1)

    del mdl, out
    torch.cuda.empty_cache(); gc.collect()

    print(f"  [{label}] Done -- LM prefill attention, {n_img} tokens, {side}x{side} grid")
    return heat

# ── Overlay + save ─────────────────────────────────────────────────────────────

def overlay(img_np, heat, alpha=0.72):
    """Attention-weighted overlay.

    Low-attention pixels keep the original image; high-attention pixels
    are progressively coloured with the jet colormap.  This avoids the
    uniform blue tint that makes base- and HIKARI-model maps look alike.
    """
    from scipy.ndimage import gaussian_filter

    # Smooth the upsampled heatmap (sigma=2 suits the ~33px blocks from 10x10 grid)
    heat_s = gaussian_filter(heat.astype(np.float32), sigma=2)

    # Percentile stretch: suppress outliers while preserving relative spread
    lo, hi = np.percentile(heat_s, [3, 97])
    heat_n = np.clip((heat_s - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    cmap = cm.get_cmap("jet")
    h_rgb = (cmap(heat_n)[:, :, :3] * 255).astype(np.float32)

    # Blend weight scales with local attention → background stays natural
    w = heat_n[:, :, np.newaxis]
    blended = w * alpha * h_rgb + (1.0 - w * alpha) * img_np.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_figures(img_np, heat_base, heat_hikari, disease):
    prefix = disease.replace(" ", "_")

    # Individual overlays (kept for reference; paper uses comparison images)
    for heat, tag in [(heat_base, "base"), (heat_hikari, "hikari")]:
        if heat is None:
            continue
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.imshow(overlay(img_np, heat))
        ax.axis("off")
        path = OUT_DIR / f"{prefix}_{tag}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── Comparison (3 panels): used directly in the paper ─────────────────
    # No suptitle — LaTeX caption handles labelling.
    # Panel titles are kept short so they remain legible at column width.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99,
                        top=0.88, bottom=0.01)

    panel_cfg = [
        (img_np,  None,        "Original",             None),
        (None,    heat_base,   "Base Qwen3-VL-8B",     "(diffuse)"),
        (None,    heat_hikari, "HIKARI (Ours)",         "(focused on lesion)"),
    ]

    for ax, (raw, heat, title, sub) in zip(axes, panel_cfg):
        if raw is not None:
            ax.imshow(raw)
        elif heat is not None:
            ax.imshow(overlay(img_np, heat))
        else:
            ax.imshow(img_np)
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="red")
        label = f"{title}\n{sub}" if sub else title
        ax.set_title(label, fontsize=8, pad=3)
        ax.axis("off")

    path = OUT_DIR / f"{prefix}_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved: {path}")

# ── Sample selection ───────────────────────────────────────────────────────────

def find_samples(n=N_SAMPLES):
    """Return up to n validation image paths per target disease."""
    with open(SPLIT_INFO) as f:
        val_paths = set(json.load(f).get("val_image_paths", []))

    found = {t: [] for t in TARGETS}
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            d_raw = row["disease"].strip().lower().replace("-", " ")
            fname = row["skincap_file_path"].strip()
            fpath = str(Path(IMG_DIR) / fname).replace("/", os.sep)
            if fpath in val_paths:
                for t in TARGETS:
                    if len(found[t]) < n and t in d_raw:
                        found[t].append(fpath)
    return {t: paths for t, paths in found.items() if paths}


def save_figures_indexed(img_np, heat_base, heat_hikari, disease, idx):
    """Save individual + comparison figures with _<idx> suffix."""
    prefix = f"{disease.replace(' ', '_')}_{idx}"

    for heat, tag in [(heat_base, "base"), (heat_hikari, "hikari")]:
        if heat is None:
            continue
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.imshow(overlay(img_np, heat))
        ax.axis("off")
        fig.savefig(OUT_DIR / f"{prefix}_{tag}.png",
                    dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.88, bottom=0.01)
    for ax, (raw, heat, title, sub) in zip(axes, [
        (img_np, None,        "Original",         None),
        (None,   heat_base,   "Base Qwen3-VL-8B", "(diffuse)"),
        (None,   heat_hikari, "HIKARI (Ours)",     "(focused on lesion)"),
    ]):
        ax.imshow(overlay(img_np, heat) if heat is not None else raw)
        ax.set_title(f"{title}\n{sub}" if sub else title, fontsize=8, pad=3)
        ax.axis("off")
    path = OUT_DIR / f"{prefix}_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    samples = find_samples()
    if not samples:
        print("ERROR: no validation samples found — check CSV / split paths")
        sys.exit(1)
    total = sum(len(v) for v in samples.values())
    print(f"Found {total} candidates across {len(samples)} diseases "
          f"(up to {N_SAMPLES} per disease)\n")

    for disease, paths in samples.items():
        for idx, img_path in enumerate(paths):
            print(f"\n{'='*60}")
            print(f"  Disease : {disease}  [{idx+1}/{len(paths)}]")
            print(f"  Image   : {img_path}")

            img_pil, img_np = load_img(img_path)
            heat_base   = extract_attention(MODEL_BASE,   img_pil, device, label="base")
            heat_hikari = extract_attention(MODEL_HIKARI, img_pil, device, label="hikari")
            save_figures_indexed(img_np, heat_base, heat_hikari, disease, idx)

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {OUT_DIR.resolve()}")
    print("\nCandidates per disease (pick the clearest for the paper):")
    for disease, paths in samples.items():
        p = disease.replace(" ", "_")
        for idx in range(len(paths)):
            print(f"  {p}_{idx}_comparison.png")

if __name__ == "__main__":
    main()
