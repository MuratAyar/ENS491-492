from typing import Dict, Any
from .base_agent import BaseAgent
import json

class StarReviewerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="CaregiverScorer",
            instructions="Evaluate the caregiver's performance in the given conversation. "
                         "Assign a score from 1 to 5 based on empathy, responsiveness, and engagement. "
                         "Return JSON output with a score and justification."
        )

    async def run(self, transcript: str, sentiment: str, responsiveness: str) -> Dict[str, Any]:
        """Assign an overall caregiver rating"""
        print("[CaregiverScorer] Evaluating caregiver performance")

        try:
            # Create caregiver scoring prompt
            prompt = (
                f"Based on the following caregiver-child conversation:\n\n{transcript}\n\n"
                f"Sentiment: {sentiment}\n"
                f"Responsiveness: {responsiveness}\n\n"
                "Assign a caregiver score from 1 to 5:\n"
                "- 5: Highly empathetic, responsive, and engaged.\n"
                "- 3-4: Neutral interactions, average responsiveness.\n"
                "- 1-2: Harsh, dismissive, or passive caregiving.\n\n"
                "Return the output in JSON format as:\n"
                '{ "caregiver_score": <1-5>, "justification": "<reasoning>" }'
            )

            response = self._query_ollama(prompt)

            # Ensure JSON format is extracted before returning
            if isinstance(response, str):
                response = self._extract_json(response)

            print(f"[CaregiverScorer] Caregiver score result: {response}")

            if "error" in response:
                return {"error": response["error"]}

            return response


        except (json.JSONDecodeError, ValueError) as e:
            print(f"[CaregiverScorer] Error processing response: {e}")
            return {"error": "Failed to evaluate caregiver performance."}
