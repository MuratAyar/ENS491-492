from typing import Dict, Any
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import logging

logger = logging.getLogger("care_monitor")  # global project logger

class CategorizerAgent:
    def __init__(self):
        self.name = "Categorizer"
        self.instructions = "Categorize caregiver-child conversation into caregiving topics."
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "facebook/bart-large-mnli"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        try:
            if self.device == "cuda":
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device, dtype=torch.float16)
                self.model = self.model.half()
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        except Exception as e:
            logger.exception("[Categorizer] Model load error, loading on CPU fallback")  # ✨ changed
            self.device = "cpu"
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to("cpu")

        self.categories = [
            "Nutrition",
            "Early Learning",
            "Health",
            "Responsive Caregiving",
            "Safety & Security"
        ]
        self.classification_pipeline = pipeline(
            "zero-shot-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        logger.debug("[Categorizer] Categorizing transcript")  # ✨ use logger instead of print
        try:
            data = json.loads(messages[-1]["content"])
            transcript = data.get("transcript", "")
            if not transcript:
                return {"error": "No transcript content for categorization."}

            text_snippet = transcript[:256]
            result = self.classification_pipeline(text_snippet, candidate_labels=self.categories)
            primary_category = result["labels"][0] if "labels" in result else "General"
            secondary_categories = result["labels"][1:3] if "labels" in result else []
            return {
                "primary_category": primary_category,
                "secondary_categories": secondary_categories
            }

        except torch.cuda.OutOfMemoryError:
            logger.exception("[Categorizer] CUDA Out of Memory! Switching to CPU...")  # ✨ changed
            self.device = "cpu"
            try:
                self.model.to("cpu")
            except Exception as e:
                logger.exception("[Categorizer] Failed to move model to CPU")  # ✨ changed
            return {"error": "Memory issue: switched to CPU. Please retry."}

        except Exception as e:
            logger.exception("[Categorizer] Failed to categorize")  # ✨ changed
            return {"error": "Failed to categorize the transcript."}
