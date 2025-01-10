from typing import Dict, Any
from .base_agent import BaseAgent
import json

class StarReviewerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="StarReviewer",
            instructions="Predict a star rating (1-5) for the review based on its sentiment and content. "
                         "Ensure the output is in a valid JSON format."
        )

    async def run(self, review: str, sentiment: str) -> Dict[str, Any]:
        """Predict a star rating for the review based on its sentiment and content."""
        try:
            # Create a prompt with JSON format enforcement
            prompt = (f"Based on the following review and its sentiment, assign an appropriate star rating "
                      f"from 1 to 5. A negative review should have a rating close to 1, and a positive review "
                      f"should have a rating close to 5. Ensure the output is a valid JSON object.\n\n"
                      f"Sentiment: {sentiment}\n"
                      f"Review: {review}\n\n"
                      f"Output only the JSON object:\n"
                      f'{{"expected_stars": <integer from 1 to 5>}}')

            # Query the model
            response = self._query_ollama(prompt)
            raw_response = response.get("response", "").strip()

            # Parse the response as JSON
            parsed_response = json.loads(raw_response)

            # Validate the parsed JSON
            predicted_stars = parsed_response.get("expected_stars")
            if isinstance(predicted_stars, int) and 1 <= predicted_stars <= 5:
                return {"expected_stars": predicted_stars}
            else:
                raise ValueError("Invalid JSON format or out-of-range value")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error processing response: {e}")
            # Fallback logic for invalid predictions
            if sentiment.lower() == "negative":
                return {"expected_stars": 1}
            elif sentiment.lower() == "neutral":
                return {"expected_stars": 3}
            elif sentiment.lower() == "positive":
                return {"expected_stars": 5}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {"expected_stars": 3}  # Default fallback
