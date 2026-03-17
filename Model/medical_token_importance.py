"""
Selective Token Supervision (STS) adapted for medical caption generation.

Computes per-token importance weights using three signals:
1. Answer-proximity (w_ans): tokens in diagnosis/recommendation sentences get higher weight
2. Medical-token detection (w_reason): clinical terminology gets higher weight
3. Surprise-based (w_surp): tokens the base model finds surprising get higher weight

Combined weight: w(t) = normalize( w_ans(t) * w_reason(t) * w_surp(t) )

Medical adaptation from SIB-TinyLoRA (arXiv:2410.10040):
- w_reason: MEDICAL_PATTERNS instead of MATH_PATTERNS
- w_ans: sentence-level weights for 3-part caption structure
  (visual description: 0.6, diagnosis/recommendation: 1.0)
- IBR: L2 regularization on LoRA adapter parameters
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import numpy as np


# ============================================================================
# MEDICAL TOKEN PATTERNS
# ============================================================================

# Disease names (top-10 target classes)
DISEASE_NAMES_PATTERNS = [
    r"psoriasis",
    r"lupus\s+erythematosus",
    r"lichen\s+planus",
    r"scleroderma",
    r"photodermatoses",
    r"sarcoidosis",
    r"melanocytic\s+nev",      # nevi / nevus
    r"squamous\s+cell\s+carcinoma",
    r"basal\s+cell\s+carcinoma",
    r"acne\s+vulgaris",
]

# Morphological / clinical terminology
MORPHOLOGY_PATTERNS = [
    r"papule|plaque|macule|nodule|vesicle|pustule|bulla|cyst|wheal",
    r"erythema|erythematous",
    r"scal(e|y|ing)|crust|desquamat",
    r"ulcer|erosion|fissure|excoriation",
    r"hyperpigment|hypopigment|depigment",
    r"lichenif|keratosis|hyperkeratosis",
    r"atrophy|fibrosis|induration",
    r"telangiectas",
    r"dermoscop|dermatoscop",
    r"melanin|melanocyte|melanoma",
    r"keratinocyte|epiderm|derm",
    r"sebaceous|follicular|perifollicular",
    r"nevus|nevi",
    r"carcinoma|malignant|benign",
    r"inflammatory|autoimmune",
    r"pruritus|pruritic",
]

# Diagnostic / recommendation keywords
DIAGNOSTIC_PATTERNS = [
    r"diagnos",          # diagnosis, diagnose, diagnosed
    r"indicat",          # indicate, indicates, indicating
    r"consistent\s+with",
    r"suggest",
    r"possibility|possible",
    r"consider",
    r"biopsy",
    r"patholog",         # pathology, pathological
    r"examina",          # examination, examine
    r"recommended|recommend",
    r"differential",
]

# Location / distribution patterns
LOCATION_PATTERNS = [
    r"extensor|flexor",
    r"dorsal|palmar|plantar",
    r"symmetr",
    r"disseminat|generalized|localized",
    r"bilateral|unilateral",
]

MEDICAL_PATTERNS = (
    DISEASE_NAMES_PATTERNS
    + MORPHOLOGY_PATTERNS
    + DIAGNOSTIC_PATTERNS
    + LOCATION_PATTERNS
)

# Keywords that signal transition to diagnosis/recommendation sentence
DIAGNOSIS_KEYWORDS = re.compile(
    r"\b(indicat|diagnos|consistent\s+with|suggest|consider|possibilit|"
    r"may\s+(be|indicate|suggest)|is\s+(a|an)\s+\w+\s+(disease|condition|disorder))\b",
    re.IGNORECASE,
)

RECOMMENDATION_KEYWORDS = re.compile(
    r"\b(recommend|suggest|advis|requir|necessary|biopsy|patholog|examina|"
    r"dermoscop|confirm|further)\b",
    re.IGNORECASE,
)


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class MedicalSTSConfig:
    """Configuration for Selective Token Supervision on medical captions."""

    # w_ans: sentence-level answer-proximity weights
    description_weight: float = 0.6    # First sentence (visual description)
    diagnosis_weight: float = 1.0      # Sentences with diagnosis keywords
    recommendation_weight: float = 1.0 # Sentences with recommendation keywords
    default_weight: float = 0.8        # Other sentences

    # w_reason: medical token detection
    gamma: float = 0.1          # Weight floor for non-medical tokens
    medical_token_weight: float = 1.0

    # Feature flags
    use_answer_proximity: bool = True
    use_medical_detection: bool = True
    use_surprise_weighting: bool = False  # Off by default (needs base model pass)

    # Normalization
    normalize_weights: bool = True

    # IBR (Information Bottleneck Regularization)
    ibr_beta: float = 0.01
    ibr_prior_variance: float = 1.0


# ============================================================================
# TOKEN IMPORTANCE SCORER
# ============================================================================

class MedicalTokenImportanceScorer:
    """
    Computes token-level importance weights for medical caption generation.

    Adapted from SIB-TinyLoRA TokenImportanceScorer with:
    - Medical patterns replacing math patterns
    - Sentence-level w_ans for 3-part caption structure
    - Optional surprise weighting (w_surp)
    """

    def __init__(
        self,
        config: MedicalSTSConfig,
        tokenizer,
    ):
        self.config = config
        self.tokenizer = tokenizer

        # Compile medical regex
        self.medical_pattern = re.compile(
            "|".join(MEDICAL_PATTERNS), re.IGNORECASE
        )

        # Build medical token ID set from vocabulary
        self._build_medical_token_set()

    def _build_medical_token_set(self):
        """Identify which token IDs correspond to medical terminology."""
        self.medical_token_ids: Set[int] = set()

        try:
            vocab = self.tokenizer.get_vocab()
        except Exception:
            # Fallback for tokenizers without get_vocab()
            vocab = {}

        for token_str, token_id in vocab.items():
            try:
                decoded = self.tokenizer.decode([token_id], skip_special_tokens=True)
                if decoded and self.medical_pattern.search(decoded):
                    self.medical_token_ids.add(token_id)
            except Exception:
                pass

        print(f"[STS] Medical token set: {len(self.medical_token_ids)} tokens identified")

    def _classify_sentence_weight(self, sentence: str, sentence_idx: int) -> float:
        """
        Assign a sentence-level weight based on its content.

        - First sentence (visual description): description_weight (0.6)
        - Sentences with diagnosis keywords: diagnosis_weight (1.0)
        - Sentences with recommendation keywords: recommendation_weight (1.0)
        - Other: default_weight (0.8)
        """
        cfg = self.config
        if RECOMMENDATION_KEYWORDS.search(sentence):
            return cfg.recommendation_weight
        if DIAGNOSIS_KEYWORDS.search(sentence):
            return cfg.diagnosis_weight
        if sentence_idx == 0:
            return cfg.description_weight
        return cfg.default_weight

    def compute_answer_proximity_weights(
        self,
        input_ids: torch.Tensor,
        response_start: int,
        response_end: int,
    ) -> torch.Tensor:
        """
        Compute sentence-level answer-proximity weights for medical captions.

        Splits the response text into sentences, assigns a weight to each
        sentence based on its clinical content, and maps back to token level.
        """
        seq_len = input_ids.shape[-1]
        weights = torch.ones(seq_len, dtype=torch.float32)

        if not self.config.use_answer_proximity:
            return weights

        try:
            # Decode response tokens
            response_ids = input_ids[response_start:response_end].tolist()
            response_text = self.tokenizer.decode(
                response_ids, skip_special_tokens=True
            )

            # Split into sentences (simple: split on '. ' or '.\n')
            sentences = re.split(r'(?<=[.!?])\s+', response_text.strip())
            sentences = [s for s in sentences if s.strip()]

            if not sentences:
                return weights

            # Assign weights per sentence
            sentence_weights = [
                self._classify_sentence_weight(s, i) for i, s in enumerate(sentences)
            ]

            # Map sentence weights back to token positions
            # by encoding each sentence cumulatively
            char_offset = 0
            token_offset = response_start

            for sent_idx, (sentence, sw) in enumerate(zip(sentences, sentence_weights)):
                # Approximate: encode the sentence to get token count
                sent_ids = self.tokenizer.encode(
                    sentence, add_special_tokens=False
                )
                sent_token_len = len(sent_ids)

                end_pos = min(token_offset + sent_token_len, response_end)
                weights[token_offset:end_pos] = sw
                token_offset = end_pos

                if token_offset >= response_end:
                    break

            # Any remaining tokens (e.g., EOS): use last sentence weight
            if token_offset < response_end:
                weights[token_offset:response_end] = sentence_weights[-1]

        except Exception:
            # Fallback: uniform weights
            weights[response_start:response_end] = self.config.default_weight

        # Zero out prompt tokens (not used for loss)
        weights[:response_start] = 0.0

        return weights

    def compute_medical_weights(
        self,
        input_ids: torch.Tensor,
        response_start: int,
        response_end: int,
    ) -> torch.Tensor:
        """
        Compute medical-token detection weights.

        w_reason(t) = 1.0 if token is medical terminology, else gamma
        """
        seq_len = input_ids.shape[-1]
        weights = torch.full(
            (seq_len,), self.config.gamma, dtype=torch.float32
        )

        if not self.config.use_medical_detection:
            return torch.ones(seq_len, dtype=torch.float32)

        for t in range(response_start, min(response_end, seq_len)):
            token_id = input_ids[t].item()
            if token_id in self.medical_token_ids:
                weights[t] = self.config.medical_token_weight

        weights[:response_start] = 0.0
        return weights

    def compute_weights(
        self,
        input_ids: torch.Tensor,
        response_start: int,
        response_end: int,
    ) -> torch.Tensor:
        """
        Compute combined token importance weights.

        w(t) = normalize( w_ans(t) * w_reason(t) )
        (w_surp omitted by default — add precompute_surprise_cache if needed)
        """
        w_ans = self.compute_answer_proximity_weights(
            input_ids, response_start, response_end
        )
        w_reason = self.compute_medical_weights(
            input_ids, response_start, response_end
        )

        weights = w_ans * w_reason

        # Zero non-response
        weights[:response_start] = 0.0
        if response_end < len(weights):
            weights[response_end:] = 0.0

        # Normalize to mean=1 over response tokens
        if self.config.normalize_weights:
            resp_w = weights[response_start:response_end]
            mean_w = resp_w.mean()
            if mean_w > 0:
                weights[response_start:response_end] = resp_w / mean_w

        return weights

    def compute_weights_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weights for a batch, using labels to find response positions.

        The `labels` tensor uses -100 for prompt tokens and actual IDs
        for response tokens. We use this to identify response_start/end.

        Args:
            input_ids: (batch_size, seq_len)
            labels:    (batch_size, seq_len) — -100 for non-response tokens

        Returns:
            weights: (batch_size, seq_len-1) — aligned with shifted labels
        """
        batch_size, seq_len = input_ids.shape
        # We'll compute weights for the shifted version (seq_len - 1)
        all_weights = torch.ones(batch_size, seq_len - 1, dtype=torch.float32)

        for i in range(batch_size):
            lbl = labels[i]  # (seq_len,)
            ids = input_ids[i]  # (seq_len,)

            # Find response span in the SHIFTED label space
            # shift_labels = labels[:, 1:] → labels[i, 1:]
            shift_lbl = lbl[1:]
            valid_mask = (shift_lbl != -100)
            valid_positions = valid_mask.nonzero(as_tuple=False).squeeze(-1)

            if valid_positions.numel() == 0:
                # No response tokens — use uniform weights
                continue

            response_start = valid_positions[0].item()
            response_end = valid_positions[-1].item() + 1  # exclusive

            # Compute weights on the original (unshifted) ids
            # The token at shift position t corresponds to input_ids[t+1]
            # For response_start in shifted space → token at input_ids[response_start+1]
            # We compute on the full ids and then take ids[1:] alignment
            # Simplification: map shifted [response_start, response_end] back to
            # original ids by adding 1
            orig_start = response_start + 1  # in input_ids space
            orig_end = response_end + 1

            w = self.compute_weights(ids, orig_start, min(orig_end, seq_len))

            # Align to shifted space: w[1:] maps to shift positions
            all_weights[i] = w[1:]

        return all_weights


# ============================================================================
# IBR: Information Bottleneck Regularization (L2 on LoRA params)
# ============================================================================

def compute_ibr_loss(
    model: nn.Module,
    beta: float = 0.01,
    prior_variance: float = 1.0,
) -> torch.Tensor:
    """
    Information Bottleneck Regularization: L2 penalty on LoRA adapter params.

    L_IBR = beta * sum(||param||^2) / (2 * prior_variance)

    Only applies to LoRA-specific parameters (lora_A, lora_B).
    """
    reg = torch.tensor(0.0, dtype=torch.float32)
    device = None

    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_A" in name or "lora_B" in name):
            if device is None:
                device = param.device
            reg = reg + torch.sum(param.float() ** 2)

    if device is not None:
        reg = reg.to(device)

    loss = beta * reg / (2.0 * prior_variance)
    return loss


# ============================================================================
# CUSTOM SFT TRAINER WITH STS LOSS
# ============================================================================

class STSSFTTrainer:
    """
    Mixin to add STS (Selective Token Supervision) + IBR loss to SFTTrainer.

    Usage:
        class MySTSTrainer(STSSFTTrainer, SFTTrainer):
            pass

        trainer = MySTSTrainer(
            sts_scorer=scorer,
            sts_config=sts_config,
            ...normal SFTTrainer args...
        )
    """

    def setup_sts(
        self,
        sts_scorer: MedicalTokenImportanceScorer,
        sts_config: MedicalSTSConfig,
    ):
        """Call after __init__ to attach STS components."""
        self._sts_scorer = sts_scorer
        self._sts_config = sts_config
        self._sts_enabled = True
        print(f"[STS] Enabled — IBR beta={sts_config.ibr_beta}, "
              f"medical detection={sts_config.use_medical_detection}, "
              f"sentence weights: desc={sts_config.description_weight}, "
              f"diag={sts_config.diagnosis_weight}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override SFTTrainer.compute_loss to apply STS + IBR."""
        if not getattr(self, "_sts_enabled", False):
            return super().compute_loss(model, inputs, return_outputs, **kwargs)

        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")

        if labels is None or input_ids is None:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)

        # Guard: Unsloth compiled training may return non-tensor logits
        if not isinstance(logits, torch.Tensor):
            # Fall back: use outputs.loss + IBR only (STS weights skipped)
            base_loss = getattr(outputs, "loss", None)
            if isinstance(base_loss, torch.Tensor):
                ibr_loss = compute_ibr_loss(
                    model,
                    beta=self._sts_config.ibr_beta,
                    prior_variance=self._sts_config.ibr_prior_variance,
                ).to(base_loss.device).to(base_loss.dtype)
                total_loss = base_loss + ibr_loss
                return (total_loss, outputs) if return_outputs else total_loss
            return super().compute_loss(model, inputs, return_outputs, **kwargs)

        B, T, V = logits.shape

        # Compute per-token cross-entropy (unweighted)
        import torch.nn.functional as F
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token_loss = loss_fct(
            shift_logits.view(-1, V), shift_labels.view(-1)
        ).view(B, T - 1)

        # Compute STS token weights
        token_weights = self._sts_scorer.compute_weights_batch(
            input_ids.cpu(), labels.cpu()
        ).to(logits.device)  # (B, T-1)

        # Weighted loss
        mask = (shift_labels != -100).float()
        denom = mask.sum().clamp(min=1.0)
        weighted_loss = (per_token_loss * token_weights * mask).sum() / denom

        # IBR regularization
        ibr_loss = compute_ibr_loss(
            model,
            beta=self._sts_config.ibr_beta,
            prior_variance=self._sts_config.ibr_prior_variance,
        )
        if ibr_loss.device != weighted_loss.device:
            ibr_loss = ibr_loss.to(weighted_loss.device)

        total_loss = weighted_loss + ibr_loss.to(weighted_loss.dtype)

        if return_outputs:
            return total_loss, outputs
        return total_loss
