import json
import logging
import re
from typing import Dict, Any, List

from agents.hf_cache import get_sarcasm_pipe

logger = logging.getLogger("care_monitor")


class SarcasmDetectionAgent:
    """
    Detects irony / sarcasm in caregiver utterances.

    Strategy:
        1. Extract caregiver lines.
        2. Run the HF model on each line (top_k=None → both labels).
        3. Keep the line with the highest P(irony).

    Returns
    -------
    {
      "sarcasm": 0-1 float               # max P(irony) seen
      "sarcasm_lines": [                 # per-utterance details
          {"text": "...", "prob_irony": 0.42, "prob_non_irony": 0.58}
      ]
    }
    """

    CAREGIVER_TAGS = ("Caregiver:", "Woman:", "Mother:", "Dad:", "Mum:")

    def __init__(self, max_chars: int = 256):
        self.pipe = get_sarcasm_pipe()
        self.max_chars = max_chars

    async def run(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            payload = messages[-1]["content"]
            parsed = json.loads(payload)
            raw = parsed.get("transcript", "")

            # 1️⃣ extract caregiver utterances
            care_lines: List[str] = []
            for line in filter(None, raw.splitlines()):
                if any(tag in line for tag in self.CAREGIVER_TAGS):
                    # strip “[00:03] ” and the speaker tag
                    clean = re.sub(r"^\s*\[\d{1,2}:\d{2}\]\s*", "", line)
                    clean = clean.split(":", 1)[-1].strip()
                    if clean:
                        care_lines.append(clean)

            # 2️⃣ run model sentence-by-sentence
            results: List[Dict[str, Any]] = []
            max_irony = 0.0
            for line in care_lines or [raw]:  # fallback to whole text
                txt = line[-self.max_chars :]
                raw_preds = self.pipe(txt, top_k=None)
                preds = raw_preds[0] if isinstance(raw_preds, list) and isinstance(raw_preds[0], list) else raw_preds
                scores = {}
                for p in preds:
                    if isinstance(p, dict) and "label" in p and "score" in p:
                        scores[p["label"].lower()] = p["score"]
                p_irony = scores.get("irony") or scores.get("label_1") or 0.0
                p_non = scores.get("non_irony") or scores.get("label_0") or 1.0 - p_irony

                results.append(
                    {
                        "text": txt,
                        "prob_irony": round(float(p_irony), 4),
                        "prob_non_irony": round(float(p_non), 4),
                    }
                )
                if p_irony > max_irony:
                    max_irony = p_irony

            return {
                "sarcasm": round(float(max_irony), 3),
                "sarcasm_lines": results,
            }

        except Exception as e:
            logger.exception(f"[SarcasmDetectionAgent] Error: {e}")
            return {
                "sarcasm": 0.0,
                "sarcasm_lines": [],
                "error": str(e),
            }
