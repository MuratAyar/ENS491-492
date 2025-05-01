import logging 
from typing import Dict, Any
from .base_agent import BaseAgent

class StarReviewerAgent(BaseAgent):
    """Agent that provides a holistic caregiver performance rating."""

    def __init__(self):
        super().__init__(
            name="CaregiverScorer",
            instructions=(
                "You are a caregiving expert evaluating the adult's behaviour in a caregiver–child dialogue. "
                "ALWAYS return STRICT JSON with the keys below:\n"
                "{ caregiver_score:int(1-5), tone:str, empathy:str, responsiveness:str, justification:str }"
            )
        )

    async def run(self, transcript: str, sentiment: str, responsiveness: str) -> Dict[str, Any]:
        prompt = f"""
        ### TASK
        Rate the caregiver’s performance **1-5** (5 = exemplary) using empathy, tone and engagement as criteria.
        Return STRICT JSON only.

        ### CONTEXT
        Overall sentiment (HF model): {sentiment}
        Responsiveness (HF heuristic): {responsiveness}

        ### CONVERSATION
        {transcript}
        """

        raw = self._query_ollama(prompt)

        # ── Dayanıklı JSON parse ──
        if isinstance(raw, str):
            raw = self._extract_json(raw)

        if "caregiver_score" not in raw:
            # LLM valid JSON döndüremediyse geriye boş bir çerçeve ver – UI çökmesin
            logging.warning("[StarReviewer] Invalid output, falling back to defaults → %s", raw)
            return dict(
                caregiver_score=0,
                tone="Unknown",
                empathy="Unknown",
                responsiveness="Unknown",
                justification="LLM returned malformed output.",
            )

        # Tip güvenliği
        raw["caregiver_score"] = int(float(raw["caregiver_score"]))
        return raw

