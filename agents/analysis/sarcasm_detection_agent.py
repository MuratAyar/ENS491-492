import json, re, logging
from typing import Dict, Any, List
from agents.hf_cache import get_sarcasm_pipe

logger = logging.getLogger("care_monitor")


class SarcasmDetectionAgent:
    """
    Detect irony / sarcasm in caregiver utterances.

    Returns
    -------
    {
      "sarcasm": float,              # max irony prob (0-1)
      "sarcasm_scores": [float, ...] # per-caregiver line
    }
    """

    CAREGIVER_TAGS = ("Caregiver:", "Woman:", "Mother:", "Dad:", "Mum:")

    def __init__(self, max_chars: int = 256):
        self.pipe = get_sarcasm_pipe()
        self.max_chars = max_chars

        id2label = getattr(self.pipe.model.config, "id2label",
                           {0: "non_irony", 1: "irony"})
        self.LBL_IRONY = next(
            (v for v in id2label.values() if v.lower() == "irony"), "irony"
        )

    # --------------------------------------------------------
    @staticmethod
    def _preprocess(txt: str) -> str:
        """CardiffNLP tweet-style normalisation."""
        out = []
        for tok in txt.split():
            if tok.startswith("@") and len(tok) > 1:
                tok = "@user"
            elif tok.startswith("http"):
                tok = "http"
            out.append(tok)
        return " ".join(out)

    # --------------------------------------------------------
    async def run(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            raw_ctx = json.loads(messages[-1]["content"])
            transcript = raw_ctx.get("transcript", "")

            # 1) caregiver satırlarını çek
            care_lines: List[str] = []
            for line in filter(None, transcript.splitlines()):
                if any(tag in line for tag in self.CAREGIVER_TAGS):
                    txt = re.sub(r"^\s*\[\d{1,2}:\d{2}\]\s*", "", line)
                    txt = txt.split(":", 1)[-1].strip()
                    if txt:
                        care_lines.append(txt)

            if not care_lines:
                care_lines = [transcript]  # fallback

            sarcasm_scores, max_irony = [], 0.0

            for line in care_lines:
                clean = self._preprocess(line)[-self.max_chars:]

                preds = self.pipe(clean, top_k=None)
                if isinstance(preds, dict):
                    preds = [preds]

                prob_irony = {p["label"]: p["score"] for p in preds}.get(
                    self.LBL_IRONY, 0.0
                )

                # Heuristic down-weight for ultra-short neutral lines
                if len(line) < 25 or len(line.split()) < 4:
                    prob_irony = min(prob_irony, 0.30)

                sarcasm_scores.append(round(prob_irony, 3))
                max_irony = max(max_irony, prob_irony)

            return {
                "sarcasm": round(max_irony, 3),
                "sarcasm_scores": sarcasm_scores,
            }

        except Exception as exc:
            logger.exception("[Sarcasm] crash")
            return {
                "sarcasm": 0.0,
                "sarcasm_scores": [],
                "error": str(exc),
            }
