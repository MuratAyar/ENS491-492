# agents/llm/response_generator_agent.py
import json, logging
from typing import Dict, Any, List

from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from .base_agent import BaseAgent

logger = logging.getLogger("care_monitor")

class ResponseGeneratorAgent(BaseAgent):
    """
    Decides if a parent-notification is warranted; if yes, generates it.
    """

    def __init__(self):
        super().__init__(
            name="ParentNotifier",
            instructions=(
                "You are an expert paediatric caregiver assistant.\n"
                "Parents will already receive this notification (send_notification=true).\n"
                "Return STRICT JSON exactly like:\n"
                '{ \"send_notification\": true,\n'
                '  \"parent_notification\": \"string (≤180 characters)\",\n'
                '  \"recommendations\": [ { \"category\": \"string\", \"description\": \"string (≤140 characters)\" } ]\n'
                '}'
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
            if ctx.get("send_notification") is False:
                return {"send_notification": False,
                        "parent_notification": "",
                        "recommendations": []}

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
            tone_sc      = ctx.get("tone", 5)
            empathy_sc   = ctx.get("empathy", 5)
            abuse_flag   = ctx.get("abuse_flag", False)

            prompt = f"""
            ### TASK
            Write one supportive paragraph for parents (parent_notification) and up to 3 short recommendations.

            ### CONTEXT METRICS
            Category               : {cat}
            Abuse flag             : {abuse_flag}
            Caregiver score (1-10) : {caregiver_sc}
            Tone / Empathy / Resp. : {tone_sc} / {empathy_sc} / {ctx.get('responsiveness',5)}
            Avg sentiment          : {sent_avg:.3f}
            Avg toxicity           : {tox_avg:.3f}
            Sarcasm avg            : {sarcasm_avg:.3f}

            ### CONVERSATION (trimmed)
            {transcript}

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
