from transformers import pipeline
import torch, re, json
from typing import Dict, Any
import logging

logger = logging.getLogger("care_monitor")

class EmotionAnalysisAgent:
    """Agent that detects emotion per sentence."""

    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            return_all_scores=False,
            device=device,
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        try:
            data = json.loads(messages[-1]["content"])
            transcript = data.get("transcript", "")
            if not transcript:
                return {"error": "No transcript for emotion analysis."}

            chunks = re.findall(r"[\(\[](\d{1,2}:\d{2}(?:\s?[AP]M)?)[])]\s*(.*)", transcript)
            emotions = []
            for t, seg in chunks:
                res = self.pipe(seg[:256])[0]
                emotions.append({"time": t, "emotion": res["label"]})
            return {"timeline_emotions": emotions}
        except Exception as e:
            logger.exception("[EmotionAnalyzer] crashed")
            return {"error": f"Emotion analysis failed: {e}"}
