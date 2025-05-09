# orchestrator.py
from __future__ import annotations
from typing import Dict, Any
import json, re, logging, asyncio
from datetime import datetime
import langdetect
import argostranslate.package, argostranslate.translate

# ────────── Agents
from agents.analysis.analyzer_agent          import AnalyzerAgent
from agents.analysis.categorizer_agent       import CategorizerAgent
from agents.analysis.toxicity_agent          import ToxicityAgent
from agents.analysis.sarcasm_detection_agent import SarcasmDetectionAgent
from agents.llm.star_reviewer_agent          import StarReviewerAgent
from agents.llm.response_generator_agent     import ResponseGeneratorAgent

logger = logging.getLogger("care_monitor")

class Orchestrator:
    """Runs all sub-agents and returns the merged context."""

    # ─────────────────────────── init
    def __init__(self) -> None:
        self.use_translation = False
        self.analyzer_agent   = AnalyzerAgent()
        self.categorizer_agent= CategorizerAgent()
        self.tox_agent        = ToxicityAgent()
        self.sarcasm_agent    = SarcasmDetectionAgent()
        self.star_agent       = StarReviewerAgent()
        self.resp_agent       = ResponseGeneratorAgent()

    # ─────────────────────────── helpers
    def set_translation_flag(self, flag: bool) -> None:
        self.use_translation = bool(flag)

    def _detect_and_translate(self, text: str) -> Dict[str, Any]:
        if not self.use_translation:
            return {"transcript": text, "original_language": "en"}

        try:
            lang = langdetect.detect(text)
        except Exception:
            return {"transcript": text, "original_language": "unknown"}

        if lang.lower() == "en":
            return {"transcript": text, "original_language": "en"}

        try:                      # cheap Argos-Translate fallback
            inst = argostranslate.translate
            src  = next((l for l in inst.get_installed_languages()
                         if l.code.startswith(lang)), None)
            tgt  = next((l for l in inst.get_installed_languages()
                         if l.code.startswith("en")), None)
            if src and tgt:
                text = src.get_translation(tgt).translate(text)
                return {"transcript": text,
                        "original_language": lang,
                        "translation_used": True}
        except Exception:
            pass
        return {"transcript": text, "original_language": lang,
                "translation_used": False}

    # ─────────────────────────── abuse decision
    @staticmethod
    def _decide_abuse(ctx: Dict[str, Any]) -> bool:
        t_max  = ctx.get("toxicity", 0.0)
        t_scores = ctx.get("toxicity_scores", 0.0)
        sent_s = ctx.get("sentiment_score", 0.0)
        irony  = ctx.get("sarcasm", 0.0)
        group  = ctx.get("category_group", "General")
        cg_score = ctx.get("caregiver_score", 0)

        t_mean = sum(t_scores) / len(t_scores) if isinstance(t_scores, list) and t_scores else 0.0

        # 1) Sert toksik konuşma
        if t_max >= 0.60:
            return True

        # 2) Orta toksisite + güçlü negatif duygu
        if t_max >= 0.45 and sent_s <= -0.5 and irony < 0.4:
            return True

        # 3) Düşük alay, yüksek negatiflik
        if t_mean >= 0.35 and sent_s <= -0.3 and irony < 0.3:
            return True

        # 4) Kritik konularda orta toksisite
        if group in {"Safety", "Health", "Discipline"} and t_max >= 0.45:
            return True

        # 5) Sarkazm + az da olsa toksisite → pasif agresif
        if irony >= 0.80 and t_max >= 0.005:   # ← burası kritik fark!
            return True

        # 6) Empati düşüklüğü + alay + CG düşükse
        if cg_score <= 3 and irony >= 0.7:
            return True

        return False


    # ─────────────────────────── main pipeline
    async def process_transcript(self, transcript: str) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {}
        try:
            # 1. language / translation
            lang_res = self._detect_and_translate(transcript)
            ctx.update({"transcript": transcript})
            ctx.update(lang_res)
            txt = ctx["transcript"]

            # 2. fast parallel agents
            tasks = [
                self.tox_agent.run       ([{"content": json.dumps({"transcript": txt})}]),
                self.analyzer_agent.run  ([{"content": json.dumps({"transcript": txt})}]),
                self.categorizer_agent.run([{"content": json.dumps({"transcript": txt})}]),
                self.sarcasm_agent.run   ([{"content": json.dumps({"transcript": txt})}]),
            ]
            tox_r, ana_r, cat_r, sar_r = await asyncio.gather(*tasks)
            for r in (tox_r, ana_r, cat_r, sar_r):
                if isinstance(r, dict):
                    ctx.update(r)

            # 3. caregiver scoring
            score_r = await self.star_agent.run(ctx)
            if isinstance(score_r, dict):
                ctx.update(score_r)

            # 4. parent notification (heavy)
            resp_r, = await asyncio.gather(
                self.resp_agent.run([{"content": json.dumps(ctx)}])
            )
            if isinstance(resp_r, dict):
                ctx.update(resp_r)

            # 5. global abuse flag with multi-signal heuristic
            ctx["abuse_flag"] = self._decide_abuse(ctx)

            # timestamp / id assignment is handled upstream
            return ctx

        except Exception as exc:
            logger.exception("[Orchestrator] crash")
            return {"error": f"Orchestrator failed: {exc}"}
