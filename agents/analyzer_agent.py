from typing import Dict, Any
from transformers import pipeline
import torch
import json
import logging

logger = logging.getLogger("care_monitor")

class AnalyzerAgent:
    def __init__(self):
        self.name = "Analyzer"
        self.instructions = "Analyze caregiver-child interaction transcript."
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        # Load sentiment analysis pipeline (CardiffNLP TweetRoBERTa)
        self.sentiment_pipeline = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
            device=0 if device_name == "cuda" else -1
        )
        # Load toxicity detection pipeline (Toxic-BERT)
        self.toxicity_pipeline = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            tokenizer="unitary/toxic-bert",
            device=0 if device_name == "cuda" else -1,
            return_all_scores=True
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        print("[Analyzer] Conducting sentiment and toxicity analysis")
        try:
            # Expect the last message content to be a JSON string with transcript
            transcript_data = json.loads(messages[-1]["content"])
            conversation = transcript_data.get("transcript", "")
            if not conversation:
                return {"error": "No transcript provided for analysis."}
            # Limit analysis to avoid extremely long input
            text_sample = conversation[:1000]

            # Sentiment analysis (Negative/Neutral/Positive)
            sentiment_result = self.sentiment_pipeline(text_sample)
            label_map = {
                "LABEL_0": "Negative",
                "LABEL_1": "Neutral",
                "LABEL_2": "Positive"
            }
            sentiment_label = label_map.get(sentiment_result[0]["label"], "Neutral")

            # Toxicity analysis (multi-label classification)
            toxic_results = self.toxicity_pipeline(text_sample)[0]
            abusive_flag = False
            for res in toxic_results:
                label = res.get("label", "").lower()
                score = res.get("score", 0.0)
                if label in {"toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"} and score > 0.7:
                    abusive_flag = True
                    break

            # Derive tone/empathy/responsiveness heuristically (could use specialized models)
            tone = "Harsh" if sentiment_label == "Negative" and abusive_flag else "Calm" if sentiment_label == "Positive" else "Neutral"
            empathy = "High" if sentiment_label == "Positive" else "Moderate" if sentiment_label == "Neutral" else "Low"
            responsiveness = "Engaged" if sentiment_label != "Negative" else "Passive"

            result = {
                "sentiment": sentiment_label,
                "tone": tone,
                "empathy": empathy,
                "responsiveness": responsiveness
            }
            if abusive_flag:
                result["abusive"] = True
            return result
        except Exception as e:
            logger.exception("[AnalyzerAgent] crashed")
            return {"error": "Failed to analyze the transcript."}
