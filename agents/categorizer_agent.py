from typing import Dict, Any
from .base_agent import BaseAgent
import json

class CategorizerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Categorizer",
            instructions="Categorize the given review into one of the categories: "
                         "Errors & Bugs, Features, Costs, Service, General Feedback."
        )
    
    async def run(self, messages: list) -> Dict[str, Any]:
        """Categorize the review"""
        print("Categorizer: Categorizing review")

        try:
            review_data = json.loads(messages[-1]["content"])  # Parse JSON string
            review = review_data.get("review", "")
            sentiment = review_data.get("analyzing_sentiment", "Neutral")
            print(f"Categorizer input review: {review} with sentiment: {sentiment}")

            if not review:
                return {"error": "No review content found to categorize."}

            # Create categorization prompt
            prompt = (
                f"Categorize the following review based on sentiment ('{sentiment}') and content: '{review}'"
            )
            categorization_result = self._query_ollama(prompt)
            print(f"Categorization result: {categorization_result}")

            # Extract category name
            category_text = categorization_result.get("response", "")
            category = "General Feedback"  # Default value
            predefined_categories = ["Errors & Bugs", "Features", "Costs", "Service", "General Feedback"]
            for predefined in predefined_categories:
                if predefined.lower() in category_text.lower():
                    category = predefined
                    break

            return {"review": review, "category": category}
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            print(f"Error categorizing review: {e}")
            return {"error": "Failed to categorize the review. Please check the input format."}

