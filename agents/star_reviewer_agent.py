from typing import Dict, Any
from .base_agent import BaseAgent

class StarReviewerAgent(BaseAgent):
    """Agent that provides a holistic caregiver performance rating."""

    def __init__(self):
        super().__init__(
            name="CaregiverScorer",
            instructions=(
                "You are a caregiving expert evaluating a caregiver's performance in a conversation with a child."
                " Always respond with a JSON object containing the requested fields."
            ),
            model="qwen:7b"
        )

    async def run(self, transcript: str, sentiment: str, responsiveness: str) -> Dict[str, Any]:
        prompt = (
            "Given the following caregiver-child conversation, evaluate the caregiver's performance.\n\n"
            f"Conversation:\n{transcript}\n\n"
            f"Known context: Sentiment = {sentiment}; Responsiveness = {responsiveness}.\n\n"
            "Assess the caregiver on empathy, responsiveness, and engagement. Provide:\n"
            "- caregiver_score: an overall performance score (1-5)\n"
            "- tone: overall tone of the caregiver (e.g., Calm, Neutral, Harsh)\n"
            "- empathy: level of empathy shown (Low, Moderate, High)\n"
            "- responsiveness: level of responsiveness (Low, Moderate, High)\n"
            "- justification: a brief explanation for the above ratings\n\n"
            "Output in JSON format with keys: caregiver_score, tone, empathy, responsiveness, justification."
        )
        res = self._query_ollama(prompt)
        if isinstance(res, dict) and "error" in res:
            return {
                "caregiver_score": "0",
                "tone": "Unknown",
                "empathy": "Unknown",
                "responsiveness": "Unknown",
                "justification": res["error"][:120],
            }
        return res
