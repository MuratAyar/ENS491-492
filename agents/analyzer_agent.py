from typing import Dict, Any
from transformers import pipeline
import torch
import json

class AnalyzerAgent:
    def __init__(self):
        self.name = "Analyzer"
        self.instructions = "Analyze caregiver-child interaction transcript."
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load BERT-based RoBERTa model for sentiment analysis
        self.sentiment_pipeline = pipeline(
            "text-classification", 
            model="cardiffnlp/twitter-roberta-base-sentiment", 
            device=0 if device == "cuda" else -1
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        print("[Analyzer] Conducting sentiment analysis")
        try:
            transcript_data = json.loads(messages[-1]["content"])
            conversation = transcript_data.get("transcript", "")
            if not conversation:
                return {"error": "No transcript provided for analysis."}

            # Run sentiment analysis (BERT-based RoBERTa)
            sentiment_result = self.sentiment_pipeline(conversation[:512])  # Limit input size
            label_map = {
                "LABEL_0": "Negative",
                "LABEL_1": "Neutral",
                "LABEL_2": "Positive"
            }
            sentiment_label = label_map.get(sentiment_result[0]["label"], "Unknown")


            return {
                "sentiment": sentiment_label,
                "tone": "Neutral",  # Default placeholder
                "empathy": "Moderate",  # Default placeholder
                "responsiveness": "Engaged"  # Default placeholder
            }

        except Exception as e:
            print(f"[Analyzer] Error analyzing transcript: {e}")
            return {"error": "Failed to analyze the transcript."}
