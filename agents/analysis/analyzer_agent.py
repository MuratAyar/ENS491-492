# AnalyzerAgent – sentence-level sentiment only
import json, logging, re
from typing import Dict, Any, List
from agents.hf_cache import get_sentiment_pipe     # ✓ yalnızca sentiment

logger = logging.getLogger("care_monitor")

class AnalyzerAgent:
    """
    Returns overall sentiment plus per-utterance sentiment scores.
    - Child ve Caregiver cümlelerinin tamamını inceler
    """

    SPEAKER_TAGS = ("Child:", "Caregiver:", "Mother:", "Dad:", "Mum:", "Woman:")

    def __init__(self, batch_size: int = 8):
        self.pipe = get_sentiment_pipe()
        self.batch = batch_size

    def _extract_lines(self, txt: str) -> List[str]:
        lines = []
        for ln in txt.splitlines():
            if any(tag in ln for tag in self.SPEAKER_TAGS):
                clean = re.sub(r"^\s*\[\d{1,2}:\d{2}\]\s*", "", ln)
                text  = clean.split(":", 1)[-1].strip()
                if text:
                    lines.append(text)
        return lines or [txt]

    async def run(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            payload = json.loads(messages[-1]["content"])
            txt     = payload.get("transcript", "")
            lines   = self._extract_lines(txt)[:128]          # güvenlik limiti

            results = self.pipe(lines, batch_size=self.batch)

            score_list = []
            for r in results:
                if isinstance(r, list):
                    r = {x['label']: x['score'] for x in r}
                else:
                    r = {r['label']: r['score']}
                pos = r.get("LABEL_2", 0.0)
                neg = r.get("LABEL_0", 0.0)
                score_list.append(round(pos - neg, 3))

            avg = sum(score_list) / len(score_list)
            overall = "Positive" if avg > 0.2 else "Negative" if avg < -0.2 else "Neutral"

            tone  = "Harsh" if overall == "Negative" else "Playful" if overall == "Positive" else "Calm"
            empathy = "High" if overall == "Positive" else "Low" if overall == "Negative" else "Moderate"
            responsiveness = "Engaged" if overall != "Negative" else "Passive"

            return {
                "sentiment": overall,
                "sentiment_score": round(avg, 3),
                "sentiment_scores": score_list,
                "tone": tone,
                "empathy": empathy,
                "responsiveness": responsiveness
            }

        except Exception as e:
            logger.exception("[AnalyzerAgent] sentiment crash")
            return {"sentiment": "Neutral", "sentiment_scores": [], "error": str(e)}
