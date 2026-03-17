"""
RAG (Retrieval-Augmented Generation) for visual few-shot disease classification.

Builds a CLIP embedding index of training images and retrieves top-K similar
images at inference time to use as few-shot examples in the prompt.

Supports:
  - RAGRetriever      : original image-only CLIP retriever (R0, backward compat)
  - HybridRAGRetriever: image + text encoder hybrid retrieval (R1-R4)
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Encoder experiment configurations
# ---------------------------------------------------------------------------

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

RAG_ENCODER_CONFIGS = {
    # R0: image-only baseline (original behavior)
    "R0": {
        "img": "openai/clip-vit-base-patch32",
        "txt": None,
        "strategy": "A",        # cross-modal (image query vs text refs via CLIP)
    },
    # R1: CLIP images + ClinicalBERT captions (two-pass: VLM description -> ClinicalBERT)
    "R1": {
        "img": "openai/clip-vit-base-patch32",
        "txt": "medicalai/ClinicalBERT",
        "strategy": "B",
    },
    # R2: SigLIP images + BGE-M3 captions (siglip-base-patch16-512 is public v1)
    "R2": {
        "img": "google/siglip-base-patch16-512",
        "txt": "BAAI/bge-m3",
        "strategy": "B",
    },
    # R3: Jina-CLIP-v2 images + MedCPT captions (Strategy B: separate spaces)
    # Note: MedCPT (768-dim) is NOT in JinaClip's shared space (1024-dim), so Strategy B
    # is required. Text scoring uses MedCPT query vs MedCPT reference embeddings.
    "R3": {
        "img": "jinaai/jina-clip-v2",
        "txt": "ncbi/MedCPT-Query-Encoder",
        "strategy": "B",
    },
    # R4: Nomic unified vision+text (shared 768-dim embedding space, cross-modal Strategy A)
    "R4": {
        "img": "nomic-ai/nomic-embed-vision-v1.5",
        "txt": "nomic-ai/nomic-embed-text-v1.5",
        "strategy": "A",
    },
}


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _embed_images_clip(model_name: str, data: List[dict], device: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """Embed images with a CLIP-compatible model. Returns (embeddings, labels, paths)."""
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    embs, labels, paths = [], [], []
    errors = 0
    for i, item in enumerate(data):
        try:
            img = Image.open(item["image_path"]).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = model.get_image_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            embs.append(feat.cpu().float().numpy().squeeze())
            labels.append(item["disease"])
            paths.append(item["image_path"])
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [RAG] Warning: embed failed for {item['image_path']}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  [RAG] Embedded {i+1}/{len(data)}...")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return np.stack(embs), labels, paths


def _embed_images_siglip(model_name: str, data: List[dict], device: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """Embed images with SigLIP-2 via AutoModel."""
    from transformers import AutoModel, AutoProcessor  # type: ignore
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    embs, labels, paths = [], [], []
    errors = 0
    for i, item in enumerate(data):
        try:
            img = Image.open(item["image_path"]).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = model.get_image_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            embs.append(feat.cpu().float().numpy().squeeze())
            labels.append(item["disease"])
            paths.append(item["image_path"])
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [RAG] Warning: embed failed: {e}")
        if (i + 1) % 500 == 0:
            print(f"  [RAG] Embedded {i+1}/{len(data)}...")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return np.stack(embs), labels, paths


def _patch_jinaclip_xattn(model) -> None:
    """Force JinaClip's EVA backbone to use standard PyTorch attention instead of xformers.
    xformers memory_efficient_attention may be unavailable even when xformers is installed
    (e.g., built for a different PyTorch/CUDA version). Standard attention gives identical
    embeddings at slightly lower memory efficiency — fine for inference."""
    patched = 0
    for mod in model.modules():
        if hasattr(mod, "xattn") and mod.xattn:
            mod.xattn = False
            patched += 1
    if patched:
        print(f"  [JinaClip] Disabled xattn in {patched} attention modules (xformers unavailable)")


def _embed_images_jinaclip(model_name: str, data: List[dict], device: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """Embed images with Jina-CLIP-v2."""
    from transformers import AutoModel  # type: ignore
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    _patch_jinaclip_xattn(model)

    embs, labels, paths = [], [], []
    errors = 0
    for i, item in enumerate(data):
        try:
            img = Image.open(item["image_path"]).convert("RGB")
            with torch.no_grad():
                feat = model.encode_image(img)
                feat = feat / np.linalg.norm(feat)
            embs.append(np.array(feat).squeeze().astype(np.float32))
            labels.append(item["disease"])
            paths.append(item["image_path"])
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [RAG] Warning: embed failed: {e}")
        if (i + 1) % 500 == 0:
            print(f"  [RAG] Embedded {i+1}/{len(data)}...")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return np.stack(embs), labels, paths


def _embed_images_qwen(model_name: str, data: List[dict], device: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """Embed images with Qwen3-VL-Embedding."""
    from transformers import AutoModel, AutoProcessor  # type: ignore
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    embs, labels, paths = [], [], []
    errors = 0
    for i, item in enumerate(data):
        try:
            img = Image.open(item["image_path"]).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = model(**inputs).image_embeds
                feat = feat / feat.norm(dim=-1, keepdim=True)
            embs.append(feat.cpu().float().numpy().squeeze())
            labels.append(item["disease"])
            paths.append(item["image_path"])
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [RAG] Warning: embed failed: {e}")
        if (i + 1) % 500 == 0:
            print(f"  [RAG] Embedded {i+1}/{len(data)}...")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return np.stack(embs), labels, paths


def _embed_images_nomic(model_name: str, data: List[dict], device: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """Embed images with Nomic embed-vision (shared space with nomic-embed-text)."""
    from transformers import AutoModel, AutoImageProcessor  # type: ignore
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_name)

    embs, labels, paths = [], [], []
    errors = 0
    for i, item in enumerate(data):
        try:
            img = Image.open(item["image_path"]).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = model(**inputs).last_hidden_state[:, 0]  # CLS token
                feat = feat / feat.norm(dim=-1, keepdim=True)
            embs.append(feat.cpu().float().numpy().squeeze())
            labels.append(item["disease"])
            paths.append(item["image_path"])
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [RAG] Warning: embed failed: {e}")
        if (i + 1) % 500 == 0:
            print(f"  [RAG] Embedded {i+1}/{len(data)}...")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return np.stack(embs), labels, paths


def _embed_images(model_name: str, data: List[dict]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Route to the right image embedding function based on model name."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[RAG] Embedding {len(data)} images with {model_name} on {device}...")

    name = model_name.lower()
    try:
        if "clip" in name and "jina" not in name:
            return _embed_images_clip(model_name, data, device)
        elif "siglip" in name:
            return _embed_images_siglip(model_name, data, device)
        elif "jina" in name:
            return _embed_images_jinaclip(model_name, data, device)
        elif "nomic" in name and "vision" in name:
            return _embed_images_nomic(model_name, data, device)
        elif "qwen" in name:
            return _embed_images_qwen(model_name, data, device)
        else:
            # Generic fallback: try AutoModel
            return _embed_images_siglip(model_name, data, device)
    except Exception as e:
        print(f"  [RAG] Failed with {model_name}: {e}. Falling back to CLIP.")
        return _embed_images_clip(CLIP_MODEL_NAME, data, device)


def _embed_texts_bert(model_name: str, texts: List[str]) -> np.ndarray:
    """Embed texts with a BERT-style model (ClinicalBERT, MedCPT, etc.)."""
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    print(f"  [RAG] Embedding {len(texts)} texts with {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()

    embs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
            # Use CLS token embedding
            feat = out.last_hidden_state[:, 0, :]
            feat = feat / feat.norm(dim=-1, keepdim=True)
        embs.append(feat.float().numpy())

    del model
    gc.collect()
    return np.concatenate(embs, axis=0)


def _embed_texts_bge(model_name: str, texts: List[str]) -> np.ndarray:
    """Embed texts with BGE-M3 via FlagEmbedding or sentence-transformers."""
    print(f"  [RAG] Embedding {len(texts)} texts with {model_name}...")
    try:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore
        model = BGEM3FlagModel(model_name, use_fp16=True)
        embs = model.encode(texts, batch_size=32, max_length=256)["dense_vecs"]
        del model
        return np.array(embs, dtype=np.float32)
    except ImportError:
        # Fallback to sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model = SentenceTransformer(model_name)
            embs = model.encode(texts, batch_size=32, normalize_embeddings=True)
            del model
            return np.array(embs, dtype=np.float32)
        except ImportError:
            print("  [RAG] Warning: FlagEmbedding and sentence_transformers not available. Skipping text embedding.")
            return None


def _embed_texts_nomic(model_name: str, texts: List[str]) -> np.ndarray:
    """Embed texts with Nomic embed-text (shared space with nomic-embed-vision)."""
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    print(f"  [RAG] Embedding {len(texts)} texts with {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()

    embs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        # Nomic text convention: prefix with task type
        batch = ["search_document: " + t for t in texts[i:i + batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        with torch.no_grad():
            feat = model(**inputs).last_hidden_state[:, 0]  # CLS token
            feat = feat / feat.norm(dim=-1, keepdim=True)
        embs.append(feat.float().numpy())

    del model
    gc.collect()
    return np.concatenate(embs, axis=0)


def _embed_texts(model_name: str, texts: List[str]) -> Optional[np.ndarray]:
    """Route to correct text embedding function."""
    if not model_name:
        return None
    name = model_name.lower()
    try:
        if "bge-m3" in name or "bge_m3" in name:
            return _embed_texts_bge(model_name, texts)
        elif "nomic" in name:
            return _embed_texts_nomic(model_name, texts)
        else:
            return _embed_texts_bert(model_name, texts)
    except Exception as e:
        print(f"  [RAG] Warning: text embedding failed ({e}). Text similarity disabled.")
        return None


# ---------------------------------------------------------------------------
# Original RAGRetriever (image-only, backward compatible)
# ---------------------------------------------------------------------------

class RAGRetriever:
    """CLIP-based visual similarity retrieval for few-shot examples."""

    def __init__(self, embeddings: np.ndarray, labels: List[str], image_paths: List[str]):
        self.embeddings = embeddings
        self.labels = labels
        self.image_paths = image_paths
        k = min(10, len(embeddings))
        self.index = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        self.index.fit(embeddings)
        self._clip_model = None
        self._clip_proc = None

    @classmethod
    def build(cls, train_data: List[dict], index_path: str = "./rag_index.npz") -> "RAGRetriever":
        """Embed all training images with CLIP and save index to disk."""
        print(f"\n[RAG] Building CLIP embedding index from {len(train_data)} training images...")
        embs, labels, paths = _embed_images(CLIP_MODEL_NAME, train_data)
        np.savez(index_path, embeddings=embs, labels=np.array(labels), paths=np.array(paths))
        print(f"[RAG] Index saved to {index_path}")
        return cls(embs, labels, paths)

    @classmethod
    def load(cls, index_path: str) -> "RAGRetriever":
        """Load a pre-built index from disk."""
        data = np.load(index_path, allow_pickle=True)
        return cls(data["embeddings"], data["labels"].tolist(), data["paths"].tolist())

    @classmethod
    def load_or_build(cls, train_data: List[dict], index_path: str) -> "RAGRetriever":
        """Load existing index if available, otherwise build from train_data."""
        if Path(index_path).exists():
            print(f"[RAG] Loading existing index from {index_path}")
            return cls.load(index_path)
        return cls.build(train_data, index_path)

    def _get_clip_encoder(self):
        """Lazy-load CLIP for query-time embedding (CPU to avoid VRAM conflict)."""
        if self._clip_model is None:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
            print("[RAG] Loading CLIP encoder for query embedding (CPU)...")
            self._clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).eval()
            self._clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        return self._clip_model, self._clip_proc

    def _embed_query(self, image: Image.Image) -> np.ndarray:
        model, processor = self._get_clip_encoder()
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.float().numpy().squeeze()

    def retrieve(self, query_image: Image.Image, k: int = 3, **kwargs) -> List[Tuple[str, str]]:
        """Return top-k (image_path, label) pairs most similar to query_image."""
        k = min(k, len(self.labels))
        query_emb = self._embed_query(query_image).reshape(1, -1)
        _, idxs = self.index.kneighbors(query_emb, n_neighbors=k)
        return [(self.image_paths[i], self.labels[i]) for i in idxs[0]]


# ---------------------------------------------------------------------------
# HybridRAGRetriever (image + text, multiple encoder combos)
# ---------------------------------------------------------------------------

class HybridRAGRetriever:
    """
    Hybrid image+text retrieval for few-shot examples.

    Strategies:
      A (cross-modal): score = alpha * cos(img_enc(q), img_emb)
                              + (1-alpha) * cos(img_enc(q), txt_emb)
        Works when the image encoder shares latent space with the text
        (CLIP, SigLIP, Jina-CLIP-v2, Qwen3-VL-Embedding).

      B (two-pass): score = alpha * cos(img_enc(q), img_emb)
                           + (1-alpha) * cos(txt_enc(description), txt_emb)
        Requires a VLM-generated text description of the query image.
        Falls back to image-only when no description is provided.
    """

    def __init__(
        self,
        img_embs: np.ndarray,
        txt_embs: Optional[np.ndarray],
        labels: List[str],
        image_paths: List[str],
        captions: List[str],
        img_encoder_name: str,
        txt_encoder_name: Optional[str],
        strategy: str = "A",
        alpha: float = 0.5,
    ):
        self.img_embs = img_embs
        self.txt_embs = txt_embs
        self.labels = labels
        self.image_paths = image_paths
        self.captions = captions
        self.img_encoder_name = img_encoder_name
        self.txt_encoder_name = txt_encoder_name
        self.strategy = strategy
        self.alpha = alpha

        k = min(10, len(img_embs))
        self.img_index = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        self.img_index.fit(img_embs)
        if txt_embs is not None:
            self.txt_index = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
            self.txt_index.fit(txt_embs)
        else:
            self.txt_index = None

        # Lazy-loaded encoders for query-time use
        self._img_model = None
        self._img_proc = None
        self._txt_model = None
        self._txt_tok = None

    # ------------------------------------------------------------------
    # Build / Load
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        data: List[dict],
        index_path: str,
        img_encoder_name: str,
        txt_encoder_name: Optional[str] = None,
        strategy: str = "A",
        alpha: float = 0.5,
    ) -> "HybridRAGRetriever":
        """Build hybrid index from data. Each item must have image_path, disease, caption."""
        print(f"\n[RAG] Building hybrid index ({img_encoder_name} + {txt_encoder_name or 'none'}) "
              f"from {len(data)} images...")

        # Embed images
        img_embs, labels, paths = _embed_images(img_encoder_name, data)

        # Embed captions if text encoder specified
        txt_embs = None
        captions = [item.get("caption", item.get("disease", "")) for item in data]
        if txt_encoder_name:
            txt_embs = _embed_texts(txt_encoder_name, captions)

        # Save index
        save_kwargs = dict(
            img_embs=img_embs,
            labels=np.array(labels),
            paths=np.array(paths),
            captions=np.array(captions),
            img_encoder=np.array([img_encoder_name]),
            txt_encoder=np.array([txt_encoder_name or ""]),
            strategy=np.array([strategy]),
            alpha=np.array([alpha]),
        )
        if txt_embs is not None:
            save_kwargs["txt_embs"] = txt_embs
            save_kwargs["has_txt"] = np.array([True])
        else:
            save_kwargs["has_txt"] = np.array([False])
        np.savez(index_path, **save_kwargs)
        print(f"[RAG] Hybrid index saved to {index_path}")

        return cls(img_embs, txt_embs, labels, paths, captions,
                   img_encoder_name, txt_encoder_name, strategy, alpha)

    @classmethod
    def load(cls, index_path: str) -> "HybridRAGRetriever":
        """Load hybrid index from disk."""
        d = np.load(index_path, allow_pickle=True)
        img_embs = d["img_embs"]
        txt_embs = d["txt_embs"] if d["has_txt"].item() else None
        labels = d["labels"].tolist()
        paths = d["paths"].tolist()
        captions = d["captions"].tolist()
        img_encoder_name = str(d["img_encoder"][0])
        txt_encoder_name = str(d["txt_encoder"][0]) or None
        strategy = str(d["strategy"][0])
        alpha = float(d["alpha"][0])
        return cls(img_embs, txt_embs, labels, paths, captions,
                   img_encoder_name, txt_encoder_name, strategy, alpha)

    @classmethod
    def load_or_build(
        cls,
        data: List[dict],
        index_path: str,
        img_encoder_name: str,
        txt_encoder_name: Optional[str] = None,
        strategy: str = "A",
        alpha: float = 0.5,
    ) -> "HybridRAGRetriever":
        if Path(index_path).exists():
            print(f"[RAG] Loading existing hybrid index from {index_path}")
            return cls.load(index_path)
        return cls.build(data, index_path, img_encoder_name, txt_encoder_name, strategy, alpha)

    # ------------------------------------------------------------------
    # Lazy encoder loading (query-time, CPU)
    # ------------------------------------------------------------------

    def _get_img_encoder(self):
        """Lazy-load image encoder for query-time embedding (CPU)."""
        if self._img_model is None:
            name = self.img_encoder_name.lower()
            if "clip" in name and "jina" not in name:
                from transformers import CLIPModel, CLIPProcessor  # type: ignore
                print(f"[RAG] Loading query image encoder: {self.img_encoder_name}")
                self._img_model = CLIPModel.from_pretrained(self.img_encoder_name).eval()
                self._img_proc = CLIPProcessor.from_pretrained(self.img_encoder_name)
            elif "siglip" in name:
                from transformers import AutoModel, AutoProcessor  # type: ignore
                self._img_model = AutoModel.from_pretrained(self.img_encoder_name).eval()
                self._img_proc = AutoProcessor.from_pretrained(self.img_encoder_name)
            elif "jina" in name:
                from transformers import AutoModel  # type: ignore
                _device = "cuda" if torch.cuda.is_available() else "cpu"
                self._img_model = AutoModel.from_pretrained(
                    self.img_encoder_name, trust_remote_code=True).to(_device).eval()
                _patch_jinaclip_xattn(self._img_model)
                self._img_proc = None
            elif "nomic" in name and "vision" in name:
                from transformers import AutoModel, AutoImageProcessor  # type: ignore
                self._img_model = AutoModel.from_pretrained(
                    self.img_encoder_name, trust_remote_code=True).eval()
                self._img_proc = AutoImageProcessor.from_pretrained(self.img_encoder_name)
            else:
                from transformers import AutoModel, AutoProcessor  # type: ignore
                self._img_model = AutoModel.from_pretrained(
                    self.img_encoder_name, trust_remote_code=True).eval()
                self._img_proc = AutoProcessor.from_pretrained(
                    self.img_encoder_name, trust_remote_code=True)
        return self._img_model, self._img_proc

    def _embed_query_image(self, image: Image.Image) -> np.ndarray:
        model, processor = self._get_img_encoder()
        name = self.img_encoder_name.lower()
        with torch.no_grad():
            if "jina" in name:
                feat = model.encode_image(image)
                feat = np.array(feat).squeeze().astype(np.float32)
                feat /= np.linalg.norm(feat) + 1e-9
                return feat
            elif "nomic" in name and "vision" in name:
                inputs = processor(images=image, return_tensors="pt")
                feat = model(**inputs).last_hidden_state[:, 0]
                feat = feat / feat.norm(dim=-1, keepdim=True)
                return feat.float().numpy().squeeze()
            else:
                inputs = processor(images=image, return_tensors="pt")
                feat = model.get_image_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                return feat.float().numpy().squeeze()

    def _get_txt_encoder(self):
        """Lazy-load text encoder for query-time text embedding (CPU)."""
        if self._txt_model is None and self.txt_encoder_name:
            name = self.txt_encoder_name.lower()
            if "bge-m3" in name or "bge_m3" in name:
                # BGE-M3 via FlagEmbedding or sentence-transformers
                try:
                    from FlagEmbedding import BGEM3FlagModel  # type: ignore
                    self._txt_model = BGEM3FlagModel(self.txt_encoder_name, use_fp16=True)
                    self._txt_tok = "flagembedding"
                except ImportError:
                    from sentence_transformers import SentenceTransformer  # type: ignore
                    self._txt_model = SentenceTransformer(self.txt_encoder_name)
                    self._txt_tok = "sentence_transformers"
            elif "nomic" in name:
                from transformers import AutoModel, AutoTokenizer  # type: ignore
                self._txt_model = AutoModel.from_pretrained(
                    self.txt_encoder_name, trust_remote_code=True).eval()
                self._txt_tok = AutoTokenizer.from_pretrained(self.txt_encoder_name)
            else:
                from transformers import AutoModel, AutoTokenizer  # type: ignore
                self._txt_model = AutoModel.from_pretrained(self.txt_encoder_name).eval()
                self._txt_tok = AutoTokenizer.from_pretrained(self.txt_encoder_name)
        return self._txt_model, self._txt_tok

    def _embed_query_text(self, text: str) -> Optional[np.ndarray]:
        if not self.txt_encoder_name:
            return None
        model, tok = self._get_txt_encoder()
        if tok == "flagembedding":
            emb = model.encode([text], max_length=256)["dense_vecs"]
            return np.array(emb[0], dtype=np.float32)
        elif tok == "sentence_transformers":
            emb = model.encode([text], normalize_embeddings=True)
            return np.array(emb[0], dtype=np.float32)
        else:
            # Nomic text: prepend task prefix for queries
            if "nomic" in self.txt_encoder_name.lower():
                text = "search_query: " + text
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = model(**inputs)
                feat = out.last_hidden_state[:, 0, :]
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.float().numpy().squeeze()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Batch cosine similarity: query (D,) vs matrix (N, D) -> (N,)"""
        q = query / (np.linalg.norm(query) + 1e-9)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
        return (matrix / norms) @ q

    def retrieve(
        self,
        query_image: Image.Image,
        k: int = 3,
        vlm_description: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """
        Return top-k (image_path, label) pairs for the query image.

        Args:
            query_image: PIL image to query
            k: number of results to return
            vlm_description: optional text description (needed for Strategy B)
        """
        k = min(k, len(self.labels))
        q_img = self._embed_query_image(query_image)

        # Image similarity scores
        img_sims = self._cosine_scores(q_img, self.img_embs)

        if self.txt_embs is not None and self.txt_index is not None:
            if self.strategy == "A":
                # Cross-modal: same image query embedding compared to text embeddings
                txt_sims = self._cosine_scores(q_img, self.txt_embs)
                scores = self.alpha * img_sims + (1.0 - self.alpha) * txt_sims

            elif self.strategy == "B" and vlm_description:
                # Two-pass: text encoder processes VLM-generated description
                q_txt = self._embed_query_text(vlm_description)
                if q_txt is not None:
                    txt_sims = self._cosine_scores(q_txt, self.txt_embs)
                    scores = self.alpha * img_sims + (1.0 - self.alpha) * txt_sims
                else:
                    scores = img_sims  # fallback: text encoding failed
            else:
                scores = img_sims  # fallback: no description for B, or txt_embs missing
        else:
            scores = img_sims

        top_k_idx = np.argsort(scores)[-k:][::-1]
        return [(self.image_paths[i], self.labels[i], self.captions[i]) for i in top_k_idx]
