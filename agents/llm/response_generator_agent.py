from typing import Dict, Any
import json

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
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
                "You are a paediatric caregiving expert. "
                "Given the analysis context, write an empathetic parent-notification and "
                "2-3 actionable recommendations. "
                "ALWAYS return STRICT JSON with the keys:\n"
                "{ parent_notification:str, recommendations:List[{category:str, description:str}] }"
            )
        )

        self.vector_store = PersistentClient(
            path="embeddings/chroma_index",
            settings=Settings(allow_reset=True),
        )
        embedding_function = OllamaEmbeddings(model="openhermes:7b-mistral-v2.5-q5_1")
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
            score5 = context.get("caregiver_score", 3)

            context_docs = self.retriever.similarity_search(category, k=2)
            best_practices = "\n".join(doc.page_content for doc in context_docs)

            # ––– FEW-SHOT GUIDANCE –––
            prompt = f"""
            ### TASK
            Write an empathic note for the parents - one short paragraph.
            Then give max 3 concise recommendations (category + 1 sentence).

            ### CAREGIVER METRICS
            Sentiment : {sentiment}
            Category  : {category}
            Score (1-5): {score5}

            ### CONVERSATION (truncated to 1000 chars)
            {transcript[:1000]}

            ### BEST PRACTICES (retrieved)
            {best_practices}

            ### OUTPUT FORMAT (STRICT JSON)
            {{"parent_notification": "...", "recommendations":[{{"category":"...","description":"..."}}]}}
            """

            raw = self._query_ollama(prompt)
            data = self._extract_json(raw if isinstance(raw, str) else json.dumps(raw))

            # -- Çıktıyı normalize et --
            data.setdefault("parent_notification", "No summary generated.")
            recs = data.get("recommendations", [])
            if isinstance(recs, str):  # tek string geldiyse listeye sar
                recs = [{"category": "General", "description": recs}]
            data["recommendations"] = [
                {"category": r.get("category", "General"), "description": r.get("description", "")}
                for r in recs
            ]
            return data

        except Exception as exc:
            logging.exception("[ParentNotifier] crashed")
            return {"error": f"ParentNotifier exception: {exc}"}
