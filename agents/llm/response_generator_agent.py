# agents/llm/response_generator_agent.py
import json, logging
from typing import Dict, Any, List

from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from .base_agent import BaseAgent

logger = logging.getLogger("care_monitor")

IMPORTANT_CATEGORIES = {
    "Feeding", "Health", "Safety", "Medication",
    "Emotions", "Accident", "Sleep-Routine", "Devices"
}

class ResponseGeneratorAgent(BaseAgent):
    """
    Decides if a parent-notification is warranted; if yes, generates it.
    """

    def __init__(self):
        super().__init__(
            name="ParentNotifier",
            instructions=(
                "You are an expert paediatric caregiver assistant.\n"
                "First decide whether parents need a push-notification.\n"
                "If NOT, return:\n"
                '{ "send_notification": false }\n'
                "If YES, return STRICT JSON exactly like this:\n"
                '{ "send_notification": true,\n'
                '  "parent_notification": "string (≤180 characters)",\n'
                '  "recommendations": [\n'
                '    { "category": "string", "description": "string (≤140 characters)" }\n'
                '  ]\n'
                '}'          # ← burada bitiyor
            ),
        )

        # optional best-practice retrieval (same as önceki kod)
        #client = PersistentClient(path="embeddings/chroma_index", settings=Settings(allow_reset=True))
        self.retriever = Chroma(
            embedding_function=OllamaEmbeddings(model="openhermes:7b-mistral-v2.5-q5_1"),
            persist_directory="embeddings/chroma_index",  # ✅ Sadece bu yeterli
        )

    # ------------------------------------------------------------------ #
    async def run(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            ctx = json.loads(messages[-1]["content"])

            # ---------------- Heuristik kararı ----------------
            def should_notify(ctx):
                cat = ctx.get("primary_category", "General")
                tox = ctx.get("toxicity", 0.0)
                sent = ctx.get("sentiment_score", 0.0)
                sarcasm = ctx.get("sarcasm", 0.0)
                cg_score = ctx.get("caregiver_score", 5)

                return any([
                    cat in IMPORTANT_CATEGORIES,
                    (tox > 0.35 and abs(sent) < 0.2),
                    (sarcasm > 0.88 and sent < 0.2),
                    (cg_score <= 5 and tox > 0.1),           # ✅ eşik düşürüldü
                    (sent < -0.7 and cg_score <= 4),         # ✅ AŞIRI NEGATİF durum
                ])

            # Hiç gerek yoksa hemen çık
            if not should_notify(ctx):
                return {
                    "send_notification": False,
                    "parent_notification": "",
                    "recommendations": []
                }

             # ---------------- LLM’e prompt --------------------
            cat = ctx.get("primary_category", "General")
            transcript = ctx.get("transcript", "")[:1200]
            tox_scores = ctx.get("toxicity_scores", [])
            sent_scores = ctx.get("sentiment_scores", [])
            sarcasm_scores = ctx.get("sarcasm_scores", [])

            # Averages
            sent_avg = ctx.get("sentiment_score", 0.0)
            tox_avg = ctx.get("toxicity", 0.0)
            sarcasm_avg = ctx.get("sarcasm", 0.0)
            caregiver_sc = ctx.get("caregiver_score", 5)

            # best-practice snippet
            best_docs = self.retriever.similarity_search(cat, k=2)
            best_practices = "\n".join(d.page_content for d in best_docs)

            prompt = f"""
            ### TASK
            Parents rely on concise, kind notifications. 1) Write ONE short paragraph
            (parent_notification). 2) Provide up to 3 recommendations
            (category + one sentence).

            ### CONTEXT METRICS
            Category               : {cat}
            Avg sentiment score    : {sent_avg:.3f}
            Sentence sentiment[]   : {sent_scores}
            Avg toxicity           : {tox_avg:.3f}
            Toxicity per sentence[]: {tox_scores}
            Avg sarcasm            : {sarcasm_avg:.3f}
            Sarcasm per sent[]     : {sarcasm_scores}
            Caregiver score (1-10) : {caregiver_sc}

            ### CONVERSATION (trimmed)
            {transcript}

            ### BEST PRACTICES
            {best_practices}

            ### STRICT OUTPUT JSON
            {{"send_notification": true,
            "parent_notification": "...",
            "recommendations":[{{"category":"...","description":"..."}}]}}
            """
            raw = self._query_ollama(prompt)

            # LLM çıktısını güvenli şekilde ayrıştır
            if isinstance(raw, dict):
                raw_str = raw.get("raw_output", "")
                if isinstance(raw_str, str) and raw_str.strip().startswith("{"):
                    data = self._extract_json(raw_str)
                else:
                    data = raw
            elif isinstance(raw, str):
                data = self._extract_json(raw)
            else:
                data = self._extract_json(str(raw))

            # Tavsiyeleri işleyip sınırla
            recs: List[Any] = data.get("recommendations", [])

            if isinstance(recs, dict):
                recs = [recs]
            elif isinstance(recs, str):
                recs = [{"category": "General", "description": recs}]

            data["recommendations"] = [
                {
                    "category": r.get("category", "General"),
                    "description": r.get("description", "")[:140],
                }
                for r in recs[:3]
            ]

            data.setdefault("parent_notification", "")
            return data
        # --------------------------------------------------------
        # Hata durumunda varsayılan değerler
        except Exception as exc:
            logger.exception("[ParentNotifier] crashed")
            return {"send_notification": False,
                    "parent_notification": "",
                    "recommendations": [],
                    "error": str(exc)}
