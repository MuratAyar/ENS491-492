from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch, re, json, logging
from typing import Dict, Any
from agents.hf_cache import get_unbiased_toxic_roberta

logger = logging.getLogger("care_monitor")

class AbuseDetectionAgent:
    """Sentence-level toxic / non-toxic flag using Unitary RoBERTa."""

    def __init__(self):
        tok, model = get_unbiased_toxic_roberta()
        self.pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tok,
            device=0 if torch.cuda.is_available() else -1,
            top_k=None,
            truncation=True,
            max_length=256,
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        try:
            ctx = json.loads(messages[-1]["content"])
            txt = ctx.get("transcript", "")
            if not txt:
                return {"error": "No transcript for abuse detection."}

            segments = re.findall(r"[\(\[](\d{1,2}:\d{2}(?:\s?[AP]M)?)[]\)]\s*(.*)", txt)
            timeline = []
            if segments:
                times = [t for t, seg in segments]
                texts = [seg[:256] for _, seg in segments]
                try:
                    results = self.pipe(texts)
                except Exception:
                    # Fallback to sequential processing if batch fails
                    results = [self.pipe(text)[0] for text in texts]
                    # The above returns a dict for each text (top label only if top_k not None, but here top_k=None so list of dicts)
                # Ensure results is list of lists of dicts (pipeline with top_k=None yields list of dict per input)
                if results and isinstance(results[0], dict):
                    # If pipeline returned top label dict for each input (in case top_k ignored), wrap them
                    results = [[res] for res in results]
                for (t, seg), res in zip(segments, results):
                    if isinstance(res, list):
                        # res is list of label-score dicts for this segment
                        lbl = max(res, key=lambda d: d["score"])["label"].lower()
                    else:
                        # res might be a single dict (if pipeline returned only one per input)
                        lbl = res.get("label", "").lower()
                    timeline.append({"time": t, "abusive": lbl.startswith("toxic")})
            return {"timeline_abuse": timeline}
        except Exception:
            logger.exception("[AbuseDetectionAgent] crashed")
            return {"error": "Abuse detection failed."}
