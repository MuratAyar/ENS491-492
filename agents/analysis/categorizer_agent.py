# agents/categorizer_agent.py
from __future__ import annotations
from typing import Dict, Any, List
import json, logging, pathlib

from agents.hf_cache import get_categorizer_pipe   # facebook/bart-large-mnli

logger = logging.getLogger("CategorizerAgent")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class CategorizerAgent:
    """
    Caregiver–child konuşmasını “alt kategori” ve “üst kategori” (category_group)
    olarak etiketler.  Zero-shot → facebook/bart-large-mnli
    """

    def __init__(self) -> None:
        self.pipe = get_categorizer_pipe()
        self.groups, self.labels, self.reverse = self._load_structure()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _load_structure():
        """
        categories.json   →  Dict[str, List[str]]
        Dönen:
            groups   : orijinal dict
            labels   : düz liste (candidate_labels)
            reverse  : alt_etiket  ➜  üst_kategori
        """
        base = pathlib.Path(__file__).parent
        path = base / "categories.json"
        with open(path, "r", encoding="utf-8") as f:
            groups: Dict[str, List[str]] = json.load(f)

        labels, reverse = [], {}
        for parent, subs in groups.items():
            for s in subs:
                labels.append(s)
                reverse[s] = parent
        return groups, labels, reverse

    # ------------------------------------------------------------------ main
    async def run(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Mesaj formatı   {"role":"user","content":"{\"transcript\":\"…\"}"}
        """
        try:
            payload = json.loads(messages[-1]["content"])
            txt = payload.get("transcript", "").strip()
            if not txt:
                return {"error": "Empty transcript"}

            snippet = txt[:512]                                  # ≈ ilk 512 karakter yeterli
            out = self.pipe(snippet, candidate_labels=self.labels, multi_label=False)

            best_label = out["labels"][0] if out["labels"] else "Uncategorized"
            group      = self.reverse.get(best_label, "General")

            # En yüksek skora sahip 2 alt etiket daha (sekonder)
            ranked = list(zip(out["labels"], out["scores"]))
            secondary = [l for l, _ in ranked[1:3] if l != best_label]

            return {
                "primary_category"   : best_label,   # örn. "Refusing to Eat"
                "category_group"     : group,        # örn. "Feeding"
                "secondary_categories": secondary    # isteğe bağlı
            }

        except Exception as e:
            logger.exception("CategorizerAgent failed")
            return {"error": str(e)}
