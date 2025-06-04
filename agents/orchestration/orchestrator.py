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
from agents.llm.should_notify_agent   import ShouldNotifyAgent


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
        self.decider_agent = ShouldNotifyAgent()


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
            
            # 4. notification DECISION (LLM)
            decide_r = await self.decider_agent.run(ctx)
            ctx["send_notification"] = decide_r.get("notify", False)
            ctx["notify_reason"]     = decide_r.get("reason", "")

            # 5. parent notification (heavy)
            if ctx["send_notification"]:
                resp_r, = await asyncio.gather(
                    self.resp_agent.run([{"content": json.dumps(ctx)}])
                )
                if isinstance(resp_r, dict):
                    ctx.update(resp_r)
            else:
                # Boş placeholder – front-end karşılığı net olsun
                ctx.update({"parent_notification": "",
                            "recommendations": []})

            # timestamp / id assignment is handled upstream
            return ctx

        except Exception as exc:
            logger.exception("[Orchestrator] crash")
            return {"error": f"Orchestrator failed: {exc}"}
