# agents/toxicity_agent.py
from transformers import pipeline
import torch, json
from typing import Dict, Any


class ToxicityAgent:
    """Flag transcripts that contain abusive language (threshold 0.80)."""

    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("text-classification",
                             model="unitary/toxic-bert",
                             device=device)

    async def run(self, msgs) -> Dict[str, Any]:
        txt = json.loads(msgs[-1]["content"]).get("transcript", "")[:512]
        if not txt:
            return {}
        score = self.pipe(txt)[0]["score"]
        return {"toxicity": round(score, 3), "abuse_flag": score > 0.80}
