from typing import Dict, Any, List
import json, logging, os

from agents.hf_cache import get_categorizer_pipe  # <- önbellekli model

logger = logging.getLogger("CategorizerAgent")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


class CategorizerAgent:
    """
    Granüler bakım konularına göre caregiver-child konuşmalarını sınıflandırır.
    Model, hf_cache.get_categorizer_pipe() üzerinden tek seferde belleğe alınır.
    """

    def __init__(self):
        self.name = "Categorizer"
        self.instructions = (
            "Categorize caregiver-child conversations into caregiving topics."
        )
        self.pipe = get_categorizer_pipe()

        # Kategori listesini yükle / fallback
        self.categories: List[str] = self._load_categories()

    # ------------------------------------------------------------------
    @staticmethod
    def _load_categories() -> List[str]:
        """
        'categories.json' varsa onu kullan; yoksa default liste döndür.
        """
        try:
            with open("categories.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return [
                "Breakfast",
                "Lunch",
                "Dinner",
                "Sleep",
                "Playtime",
                "Storytelling",
                "TV",
                "Hygiene",
                "Discipline",
                "Emotional Support",
                "Encouragement",
                "Instruction",
                "Caregiver Stress",
                "Crying",
                "Silence",
                "Yelling",
                "Danger",
                "Health",
                "Learning",
                "Family",
                "Outdoor",
            ]

    # ------------------------------------------------------------------
    async def run(self, messages: list[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Beklenen mesaj formatı:
        {
          "role": "user",
          "content": "{\"transcript\": \"...\"}"
        }
        """
        try:
            data = json.loads(messages[-1]["content"])
            transcript = data.get("transcript", "")
            if not transcript.strip():
                return {"error": "No transcript content for categorization."}

            # Zero-shot sınıflandırma (ilk 512 karakter yeterli)
            snippet = transcript[:512]
            result = self.pipe(
                snippet,
                candidate_labels=self.categories,
                multi_label=True,
            )

            # Sonuçları puana göre sırala
            scored = sorted(
                zip(result["labels"], result["scores"]),
                key=lambda x: x[1],
                reverse=True,
            )

            primary = scored[0][0] if scored else "Uncategorized"
            secondary = [lbl for lbl, _ in scored[1:3]] if len(scored) > 2 else []

            return {
                "primary_category": primary,
                "secondary_categories": secondary,
            }

        except Exception as exc:
            logger.exception("[Categorizer] Categorization failed.")
            return {"error": "Categorization error.", "detail": str(exc)}
