# agents/hf_cache.py
"""
Lazy-initialised, GPU–aware HuggingFace pipelines & models.

Usage:
    from agents.hf_cache import get_sentiment_pipe, get_toxicity_pipe, ...
"""

from functools import lru_cache
import torch
from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# helpers
_DEVICE = 0 if torch.cuda.is_available() else -1


@lru_cache(maxsize=1)
def get_sentiment_pipe():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
        device=_DEVICE,
    )


@lru_cache(maxsize=1)
def get_toxicity_pipe():
    return pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        tokenizer="unitary/toxic-bert",
        device=_DEVICE,
        top_k=None,
    )


@lru_cache(maxsize=1)
def get_sarcasm_pipe():
    """
    Sarcasm / irony detector → Cardiff NLP RoBERTa.
    Labels:  'irony' / 'non_irony'
    """
    model_id = "cardiffnlp/twitter-roberta-base-irony"
    try:
        mdl = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            use_safetensors=False
        ).to("cuda" if _DEVICE == 0 else "cpu")
        tok = AutoTokenizer.from_pretrained(model_id)
        # ⚠ We do *not* set return_all_scores here; we’ll ask for all scores
        #    when we call the pipeline (top_k=None).
        return pipeline(
            "text-classification",
            model=mdl,
            tokenizer=tok,
            device=_DEVICE,
        )
    except Exception as e:
        print(f"[SarcasmPipe] Fallback, model load failed: {e}")
        return lambda txt, **_: [[{"label": "non_irony", "score": 1.0},
                                  {"label": "irony",      "score": 0.0}]]

# -------------------------  CATEGORIZER  ------------------------------
@lru_cache(maxsize=1)
def get_categorizer_pipe():
    """
    Zero-shot konu sınıflandırıcısı → facebook/bart-large-mnli.
    Çok kez çağrılsa da yalnızca tek seferde belleğe yüklenir.
    """
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        tokenizer="facebook/bart-large-mnli",
        device=_DEVICE,
    )