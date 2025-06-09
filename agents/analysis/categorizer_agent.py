# agents/categorizer_agent.py
from __future__ import annotations
"""
CategorizerAgent
----------------
Classifies a caregiver–child transcript into
  • primary_category      – fine label   (e.g. "Breakfast")
  • category_group        – top bucket   (e.g. "Meals")
  • secondary_categories  – next 2 labels

Compatible with the *new* categories.json schema:
{
  "Meals": { "merge_window_min": 30, "items": ["Breakfast", ...] },
  ...
}
An old list-only JSON will also load for backwards compatibility.
"""
from typing import Dict, Any, List, Tuple
import json, logging, pathlib
from agents.hf_cache import get_categorizer_pipe  # facebook/bart-large-mnli

logger = logging.getLogger("CategorizerAgent")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class CategorizerAgent:
    # ───────────────────────────────────────────────
    def __init__(self) -> None:
        self.pipe = get_categorizer_pipe()
        self.groups, self.labels, self.reverse = self._load_structure()

    # ─────────────────────────────────────────────── helpers
    @staticmethod
    def _load_structure() -> Tuple[Dict[str, List[str]], List[str], Dict[str, str]]:
        base = pathlib.Path(__file__).parent
        path = base / "categories.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        groups: Dict[str, List[str]] = {}
        for parent, meta in raw.items():
            # new format: {"items":[...]}
            if isinstance(meta, dict) and "items" in meta:
                groups[parent] = list(meta["items"])
            # old format: ["Breakfast", "Lunch", ...]
            elif isinstance(meta, list):
                groups[parent] = list(meta)
            else:
                logger.warning("Unexpected schema for '%s' → skipped", parent)

        labels, reverse = [], {}
        for parent, subs in groups.items():
            for s in subs:
                labels.append(s)
                reverse[s] = parent

        if not labels:
            raise ValueError("categories.json produced zero labels!")
        return groups, labels, reverse

    # ─────────────────────────────────────────────── main
    async def run(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        messages[-1]["content"] == JSON string with a `transcript` field.
        """
        try:
            payload = json.loads(messages[-1]["content"])
            txt = payload.get("transcript", "").strip()
            if not txt:
                return {"error": "Empty transcript"}

            snippet = txt[:512]  # safety
            out = self.pipe(snippet, candidate_labels=self.labels, multi_label=False)

            best_label = out["labels"][0] if out.get("labels") else "Uncategorised"
            group = self.reverse.get(best_label, "General")

            ranked = list(zip(out.get("labels", []), out.get("scores", [])))
            secondary = [l for l, _ in ranked[1:3] if l != best_label]

            return {
                "primary_category": best_label,
                "category_group": group,
                "secondary_categories": secondary,
            }

        except Exception as e:
            logger.exception("CategorizerAgent failed")
            return {"error": str(e)}
