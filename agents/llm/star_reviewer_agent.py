# agents/llm/star_reviewer_agent.py
import logging, json
from typing import Dict, Any
from .base_agent import BaseAgent

logger = logging.getLogger("care_monitor")

class StarReviewerAgent(BaseAgent):
    """
    Holistic caregiver reviewer.
    • caregiver_score, tone, empathy, responsiveness  ⇒ 1-10
    • justification                                 ⇒ açıklama
    """

    def __init__(self):
        super().__init__(
            name="CaregiverScorer",
            instructions=(
                "You are a child-development expert. "
                "Given a full analysis context, rate the caregiver on a 1-10 scale.\n"
                "Return STRICT JSON with keys:\n"
                "{ caregiver_score:int(1-10), tone:int(1-10), empathy:int(1-10), "
                "responsiveness:int(1-10), justification:str }"
            )
        )

    # ------------------------------------------------------------------ #
    async def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """ctx = orchestrator’ın topladığı tam analiz sözlüğü"""
        try:
            # Prompta dâhil edeceğimiz veriler
            tx          = ctx.get("transcript", "")[:2000]
            tox_avg     = ctx.get("toxicity", 0.0)
            tox_scores  = ctx.get("toxicity_scores", [])
            sent_avg    = ctx.get("sentiment_score", 0.0)
            sent_scores = ctx.get("sentiment_scores", [])
            sarcasm_avg = ctx.get("sarcasm", 0.0)
            sarcasm_sc  = ctx.get("sarcasm_scores", [])
            category    = ctx.get("primary_category", "Unknown")

            prompt = f"""
            ### TASK
            Evaluate the ADULT caregiver’s overall performance on a **1-10** scale
            (10 = outstanding).  Also give 1-10 sub-scores for tone, empathy and
            responsiveness.  Base your judgement ONLY on the numbers & dialogue below.
            Return STRICT JSON – no extra keys.

            ### NUMERICAL CONTEXT
            Primary topic: {category}
            Avg sentiment score: {sent_avg:.3f}
            Sentence sentiments[]: {sent_scores}
            Avg toxicity: {tox_avg:.3f}
            Toxicity per Caregiver sentence[]: {tox_scores}
            Avg sarcasm: {sarcasm_avg:.3f}
            Sarcasm per Caregiver sentence[]: {sarcasm_sc}

            ### CONVERSATION (truncated)
            {tx}

            ### OUTPUT FORMAT
            {{
            "caregiver_score": 1-10,
            "tone": 1-10,
            "empathy": 1-10,
            "responsiveness": 1-10,
            "justification": "..."
            }}
            """
            raw = self._query_ollama(prompt)
            if isinstance(raw, str):
                raw = self._extract_json(raw)

            # güvenlik: zorunlu alanlar + int(1-10)
            def _clamp(v):  # type: ignore
                try:
                    v = int(round(float(v)))
                    return max(1, min(10, v))
                except Exception:
                    return 0

            keys = ["caregiver_score", "tone", "empathy", "responsiveness"]
            out = {k: _clamp(raw.get(k, 0)) for k in keys}
            out["justification"] = raw.get("justification", "No explanation.")
            return out

        except Exception as e:
            logger.exception("[StarReviewer] crash")
            return {
                "caregiver_score": 0, "tone": 0, "empathy": 0,
                "responsiveness": 0, "justification": f"Error: {e}"
            }
