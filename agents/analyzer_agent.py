from typing import Dict, Any
from .base_agent import BaseAgent
from datetime import datetime
import json


class AnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Analyzer",
            instructions="Analyze the sentiment and extract key insights from the review."
        )
    
    async def run(self, messages: list) -> Dict[str, Any]:
        """Analyze the review"""
        print("Analyzer: Conducting initial analyzing")

        try:
            workflow_context = json.loads(messages[-1]["content"])
            print(f"Workflow context: {workflow_context}")

            # Query the model
            analyzer_result = self._query_ollama(json.dumps(workflow_context))
            print(f"Analyzer result: {analyzer_result}")

            if "error" in analyzer_result:
                return {"error": analyzer_result["error"]}

            # Extract sentiment from the response
            sentiment = "Neutral"
            response_text = analyzer_result.get("response", "")
            if "**Sentiment:**" in response_text:
                start = response_text.find("**Sentiment:**") + len("**Sentiment:**")
                end = response_text.find("\n", start)
                raw_sentiment = response_text[start:end].strip()
                sentiment = "Positive" if "positive" in raw_sentiment.lower() else "Negative" if "negative" in raw_sentiment.lower() else "Neutral"

            return {
                "analyzing_report": response_text,
                "analyzing_sentiment": sentiment,
            }
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error analyzing review: {e}")
            return {"error": "Failed to analyze the review. Please check the input format."}




        
    