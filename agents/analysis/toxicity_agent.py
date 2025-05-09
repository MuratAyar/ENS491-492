# agents/analysis/toxicity_agent.py
from typing import Dict, Any, List
import json, re
from agents.hf_cache import get_toxicity_pipe

class ToxicityAgent:
    """
    Returns toxicity score for EACH caregiver utterance.
    """
    CAREGIVER_TAGS = ("Caregiver:", "Mother:", "Woman:", "Dad:", "Mum:")

    def __init__(self):
        self.pipe = get_toxicity_pipe()

    def _caregiver_lines(self, text: str) -> List[str]:
        lines = []
        for ln in text.splitlines():
            if any(tag in ln for tag in self.CAREGIVER_TAGS):
                # “[00:04] Caregiver:” → “Oh, perfect…”
                ln = re.sub(r"^\s*\[\d{1,2}:\d{2}\]\s*", "", ln)
                ln = ln.split(":", 1)[-1].strip()
                lines.append(ln)
        return lines or [text]           # fallback

    async def run(self, msgs) -> Dict[str, Any]:
        data = json.loads(msgs[-1]["content"])
        txt  = data.get("transcript", "")[:2048]
        sarcasm = float(data.get("sarcasm", 0.0))

        care_lines = self._caregiver_lines(txt)
        preds = self.pipe(care_lines, top_k=None)
        scores = [max(p, key=lambda x: x["score"])["score"] for p in preds]

        tox_max, tox_mean = max(scores), sum(scores)/len(scores)

        return {
            "toxicity_scores": [round(s, 3) for s in scores],
            "toxicity": round(tox_max, 3),
        }