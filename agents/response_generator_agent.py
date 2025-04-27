from typing import Dict, Any
import json

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from chromadb import PersistentClient
from chromadb.config import Settings

from .base_agent import BaseAgent

import logging

logger = logging.getLogger("care_monitor")  # global project logger


class ResponseGeneratorAgent(BaseAgent):
    """Generate a parent-friendly summary plus improvement suggestions."""

    def __init__(self) -> None:
        super().__init__(
            name="ParentNotifier",
            instructions=(
                "You are a caregiving expert. ALWAYS return STRICT JSON with:\n"
                "{ parent_notification: string, recommendations: list of {category, description} }\n"
                "Example:\n"
                "{\n"
                "\"parent_notification\": \"The caregiver was empathetic and responsive.\",\n"
                "\"recommendations\": [\n"
                "  {\"category\": \"Nutrition\", \"description\": \"Offer more food variety.\"},\n"
                "  {\"category\": \"Emotional Support\", \"description\": \"Praise the child's achievements.\"}\n"
                "]\n"
                "}"
            ),
            model="qwen:7b",
        )

        self.vector_store = PersistentClient(
            path="embeddings/chroma_index",
            settings=Settings(allow_reset=True),
        )
        embedding_function = OllamaEmbeddings(model="llama3.1")
        self.retriever = Chroma(
            client=self.vector_store,
            embedding_function=embedding_function,
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        print("[ParentNotifier] Generating caregiver performance summary...")
        try:
            context = json.loads(messages[-1]["content"])
            transcript = context.get("transcript", "")
            if not transcript:
                return {"error": "No transcript provided."}

            sentiment = context.get("sentiment", "Neutral")
            category = context.get("primary_category", "General")
            caregiver_score = context.get("caregiver_score", 3)
            feedback = context.get("feedback", "")

            context_docs = self.retriever.similarity_search(category, k=2)
            context_text = "\n".join(doc.page_content for doc in context_docs)

            # ––– FEW-SHOT GUIDANCE –––
            prompt = f"""
You are a paediatric-care expert.
Return STRICT JSON with keys:
  parent_notification : str
  recommendations     : list[{{category:str, description:str}}]

### GOOD EXAMPLE
Conversation:
(10:00) Child: I’m hungry.
(10:01) Caregiver: Sure – let’s grab an apple.

JSON:
{{
 "parent_notification":"The caregiver reacted promptly and kindly to the child's need.",
 "recommendations":[{{"category":"Nutrition","description":"Offer the child water too to keep them hydrated."}}]
}}

### BAD EXAMPLE
Conversation:
(11:00) Child: I'm hungry.
(11:01) Caregiver: Stop whining. You'll eat when I say so.

JSON:
{{
 "parent_notification":"The caregiver used harsh language and dismissed the child's basic need.",
 "recommendations":[
   {{"category":"Emotional Support","description":"Use calm words instead of insults."}},
   {{"category":"Nutrition","description":"Offer a healthy snack without delay."}}
 ]}}

### ACTUAL CONVERSATION
{transcript}

Caregiver metrics:
  Sentiment  : {sentiment}
  Category   : {category}
  Score      : {caregiver_score}/5
  Feedback   : {feedback}

Relevant best practices:
{context_text}

Respond with JSON only – no extra text.
"""

            raw = self._query_ollama(prompt)

            if isinstance(raw, dict) and "error" in raw:
                print(f"[ParentNotifier] LLM error response: {raw['error']}")
                return {
                    "parent_notification": "LLM error: " + raw["error"][:120],
                    "recommendations": [],
                }

            if isinstance(raw, str):
                parsed = self._extract_json(raw)
            else:
                parsed = raw

            parsed.setdefault("parent_notification", "No note provided.")
            parsed.setdefault("recommendations", [])

            if parsed["recommendations"] and isinstance(parsed["recommendations"][0], str):
                parsed["recommendations"] = [
                    {"category": "General", "description": rec}
                    for rec in parsed["recommendations"]
                ]

            return parsed

        except Exception as e:
           logger.exception("[ParentNotifier] Crashed") 
           return {"error": f"Exception during parent note generation: {e}"}
