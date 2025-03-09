from typing import Dict, Any
from .base_agent import BaseAgent
import json

class CategorizerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Categorizer",
            instructions="Categorize the given caregiver-child conversation into one or more of the categories: "
                         "Nutrition, Early Learning, Health, Responsive Caregiving, and Safety & Security. "
                         "Return JSON output with primary and secondary categories."
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        """Categorize the conversation"""
        print("[Categorizer] Categorizing transcript")

        try:
            transcript_data = json.loads(messages[-1]["content"])
            transcript = transcript_data.get("transcript", "")
            if not transcript:
                return {"error": "No transcript content found for categorization."}

            # Create categorization prompt
            prompt = (
                f"Analyze the following caregiver-child conversation:\n\n{transcript}\n\n"
                "Determine the primary caregiving category:\n"
                "- Nutrition\n"
                "- Early Learning\n"
                "- Health\n"
                "- Responsive Caregiving\n"
                "- Safety & Security\n\n"
                "If the conversation contains multiple caregiving aspects, list secondary categories as well.\n\n"
                "Return the output in JSON format as:\n"
                '{ "primary_category": "<Category>", '
                '"secondary_categories": ["<Category1>", "<Category2>"] }'
            )

            categorization_result = self._query_ollama(prompt)
            print(f"[Categorizer] Categorization result: {categorization_result}")

            if "error" in categorization_result:
                return {"error": categorization_result["error"]}

            return categorization_result
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            print(f"[Categorizer] Error categorizing transcript: {e}")
            return {"error": "Failed to categorize the transcript. Please check the input format."}
