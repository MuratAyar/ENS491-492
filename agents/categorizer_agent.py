from typing import Dict, Any
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import json

class CategorizerAgent:
    def __init__(self):
        self.name = "Categorizer"
        self.instructions = "Categorize caregiver-child conversation into caregiving topics."
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model with FP16 precision for lower VRAM usage
        self.model_name = "facebook/bart-large-mnli"

        # Load tokenizer and model separately to apply FP16 optimization
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device, dtype=torch.float16)

        # Define caregiving categories
        self.categories = [
            "Nutrition",
            "Early Learning",
            "Health",
            "Responsive Caregiving",
            "Safety & Security"
        ]

        # Create the classification pipeline manually with optimizations
        self.classification_pipeline = pipeline(
            "zero-shot-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        print("[Categorizer] Categorizing transcript")
        try:
            transcript_data = json.loads(messages[-1]["content"])
            transcript = transcript_data.get("transcript", "")

            if not transcript:
                return {"error": "No transcript content found for categorization."}

            # Limit input size to prevent memory overflow
            transcript = transcript[:256]  # Reducing token size

            # Run caregiving category classification
            classification_result = self.classification_pipeline(transcript, candidate_labels=self.categories)

            # Assign primary and secondary categories
            primary_category = classification_result["labels"][0]
            secondary_categories = classification_result["labels"][1:3]  # Top 2 secondary categories

            return {
                "primary_category": primary_category,
                "secondary_categories": secondary_categories
            }
        except torch.cuda.OutOfMemoryError:
            print("[Categorizer] CUDA Out of Memory! Switching to CPU...")
            self.device = "cpu"
            self.model.to(self.device)  # Move model to CPU if GPU runs out of memory

            return {"error": "CUDA memory issue. Switched to CPU processing."}

        except Exception as e:
            print(f"[Categorizer] Error categorizing transcript: {e}")
            return {"error": "Failed to categorize the transcript."}
