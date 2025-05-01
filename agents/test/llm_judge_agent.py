# agents/llm_judge_agent.py

import json
from typing import Any, Dict, List
from ..llm.base_agent import BaseAgent

class LLMEvaluatorAgent(BaseAgent):
    """
    Uses Qwen (or any configured BaseAgent model) to critique all aspects of the analysis:
      - overall sentiment
      - primary category
      - caregiver_score & justification
      - parent_notification
      - recommendations

    Always returns STRICT JSON with fields:
      sentiment_feedback: str
      category_feedback: str
      justification_feedback: str
      parent_notification_feedback: str
      recommendations_feedback: List[str]
    """

    def __init__(self):
        super().__init__(
            name="LLMJudge",
            instructions=(
                "You are a caregiving expert and evaluator. "
                "Given the analysis context below, provide a JSON object "
                "with detailed feedback on each part of the analysis:\n"
                "- sentiment_feedback: comments on the overall sentiment label\n"
                "- category_feedback: comments on the primary category\n"
                "- justification_feedback: critique the justification text\n"
                "- parent_notification_feedback: critique the parent notification\n"
                "- recommendations_feedback: list of critiques, one per recommendation\n\n"
                "Respond with JSON only, no extra text."
            ),
            model="qwen:7b"
        )

    async def run(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Expect last message content is JSON-encoded ctx
        ctx = json.loads(messages[-1]["content"])

        # Build prompt embedding key outputs
        prompt = (
            f"Analysis context:\n"
            f"- Sentiment: {ctx.get('sentiment')}\n"
            f"- Category: {ctx.get('primary_category')}\n"
            f"- Caregiver Score: {ctx.get('caregiver_score')}/10\n"
            f"- Justification: \"{ctx.get('justification')}\"\n"
            f"- Parent Notification: \"{ctx.get('parent_notification')}\"\n"
            f"- Recommendations:\n"
        )
        for rec in ctx.get("recommendations", []):
            prompt += f"  * {rec['category']}: {rec['description']}\n"
        prompt += "\nProvide your JSON feedback now."

        return self._query_ollama(prompt)
