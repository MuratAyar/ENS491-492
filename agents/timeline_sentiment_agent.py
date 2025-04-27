# agents/timeline_sentiment_agent.py
from transformers import pipeline
import torch, re, json
from typing import Dict, Any


class TimelineSentimentAgent:
    """Compute sentiment per time-chunk line in the transcript."""

    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("text-classification",
                             model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                             device=device)

    async def run(self, msgs) -> Dict[str, Any]:
        txt = json.loads(msgs[-1]["content"]).get("transcript", "")
        chunks = re.findall(r"[\(\[](\d{1,2}:\d{2}(?:\s?[AP]M)?)[])]\s*(.*)", txt)
        out = []
        for t, seg in chunks:
            label = self.pipe(seg[:256])[0]["label"].replace("LABEL_", "")
            out.append({"time": t, "sentiment": label})
        return {"timeline_sentiment": out}
