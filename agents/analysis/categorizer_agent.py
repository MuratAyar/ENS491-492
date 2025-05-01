from typing import Dict, Any
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import logging

# Configure logger
logger = logging.getLogger("CategorizerAgent")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
import os

class CategorizerAgent:
    def __init__(self):
        self.name = "Categorizer"
        self.instructions = "Categorize caregiver-child conversations into granular caregiving topics."
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "facebook/bart-large-mnli"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()

        self.classification_pipeline = pipeline(
            "zero-shot-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

        # Load expanded categories from file or define inline
        self.categories = self.load_categories()

    def load_categories(self):
        try:
            with open("categories.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return [
                "Breakfast", "Lunch", "Dinner", "Sleep", "Playtime", "Storytelling", "TV",
                "Hygiene", "Discipline", "Emotional Support", "Encouragement", "Instruction",
                "Caregiver Stress", "Crying", "Silence", "Yelling", "Danger", "Health",
                "Learning", "Family", "Outdoor"
            ]

    async def run(self, messages: list) -> Dict[str, Any]:
        try:
            data = json.loads(messages[-1]["content"])
            transcript = data.get("transcript", "")
            if not transcript.strip():
                return {"error": "No transcript content for categorization."}

            text_snippet = transcript[:512]
            result = self.classification_pipeline(text_snippet, candidate_labels=self.categories, multi_label=True)

            top_results = list(zip(result["labels"], result["scores"]))
            top_results = sorted(top_results, key=lambda x: x[1], reverse=True)
            primary_category = top_results[0][0] if top_results else "Uncategorized"
            secondary = [label for label, score in top_results[1:3]] if len(top_results) > 2 else []

            return {
                "primary_category": primary_category,
                "secondary_categories": secondary
            }

        except Exception as e:
            logger.exception("[Categorizer] Categorization failed.")
            return {"error": "Categorization error."}
