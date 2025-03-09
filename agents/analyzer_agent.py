from typing import Dict, Any
from .base_agent import BaseAgent
import json

class AnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Analyzer",
            instructions="Analyze the caregiver-child interaction transcript. "
                         "Evaluate sentiment, caregiver tone, and responsiveness. "
                         "Provide structured feedback with a JSON output."
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        """Analyze the caregiver-child interaction"""
        print("[Analyzer] Conducting sentiment and interaction analysis")

        try:
            transcript_data = json.loads(messages[-1]["content"])
            conversation = transcript_data.get("transcript", "")
            if not conversation:
                return {"error": "No transcript provided for analysis."}

            # Create analysis prompt
            prompt = (
                f"Analyze the following caregiver-child conversation:\n\n{conversation}\n\n"
                "Identify caregiver sentiment (Positive, Neutral, Negative), tone, empathy level, "
                "and responsiveness. Provide feedback on caregiver behavior.\n\n"
                "Return the output in JSON format as:\n"
                '{ "sentiment": "<Positive/Neutral/Negative>", '
                '"tone": "<tone description>", '
                '"empathy": "<high/medium/low>", '
                '"responsiveness": "<engaged/passive/dismissive>", '
                '"feedback": "<caregiver performance summary>" }'
            )

            # Query Llama model
            analysis_result = self._query_ollama(prompt)
            print(f"[Analyzer] Analysis result: {analysis_result}")

            if "error" in analysis_result:
                return {"error": analysis_result["error"]}

            return analysis_result
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"[Analyzer] Error analyzing transcript: {e}")
            return {"error": "Failed to analyze the transcript. Please check the input format."}
