from typing import Dict, Any
import json, re, logging, asyncio
from datetime import datetime
import langdetect
import argostranslate.package, argostranslate.translate

# Import agents
from agents.llm.response_generator_agent import ResponseGeneratorAgent
from agents.llm.star_reviewer_agent import StarReviewerAgent

from agents.analysis.analyzer_agent import AnalyzerAgent
from agents.analysis.categorizer_agent import CategorizerAgent
from agents.analysis.toxicity_agent import ToxicityAgent
from agents.analysis.abuse_detection_agent import AbuseDetectionAgent
from agents.analysis.sarcasm_detection_agent import SarcasmDetectionAgent

# Setup logger
logger = logging.getLogger("care_monitor")

class Orchestrator:
    """Runs all sub-agents in sequence and returns a combined result."""
    
    def __init__(self):
        self.use_translation = False
        self.analyzer_agent = AnalyzerAgent()
        self.categorizer_agent = CategorizerAgent()
        self.star_reviewer_agent = StarReviewerAgent()
        self.response_generator_agent = ResponseGeneratorAgent()
        self.tox_agent = ToxicityAgent()
        self.abuse_agent = AbuseDetectionAgent()
        self.sarcasm_agent= SarcasmDetectionAgent()

    # ------------------------------------------------------------------
    def set_translation_flag(self, flag: bool):
        """Enable/disable Argos-Translate on the fly (called from Streamlit)."""
        self.use_translation = bool(flag)

     # ------------------------------------------------------------------
    def _detect_and_translate(self, text: str) -> Dict[str, Any]:
         """Run ONLY if self.use_translation is True."""
         if not self.use_translation:
             return {"transcript": text, "original_language": "en"}
         try:
             lang = langdetect.detect(text)
         except Exception:
             return {"transcript": text, "original_language": "unknown"}
         if lang.lower() == "en":
             return {"transcript": text, "original_language": "en"}
         # try cheap ArgosTranslate lookup
         try:
             inst = argostranslate.translate
             src = next((l for l in inst.get_installed_languages()
                         if l.code.startswith(lang)), None)
             tgt = next((l for l in inst.get_installed_languages()
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

    async def process_transcript(self, transcript: str) -> Dict[str, Any]:
        """Process the input transcript through all analysis agents."""
        ctx: Dict[str, Any] = {}
        try:
            # 1. Language detection and optional translation
            lang_result = self._detect_and_translate(transcript)
            logger.debug(f"[Orchestrator] LanguageSwitchAgent → {lang_result}")
            if "error" in lang_result:
                print(f"[Orchestrator] Language detection error: {lang_result['error']}")
            ctx.update({"transcript": transcript})
            ctx.update(lang_result)
            transcript_to_analyze = ctx["transcript"]

            # 2-4. Parallel quick analysis: toxicity, sentiment & tone, topic categorization
            tasks: list = [
                self.tox_agent.run([{"content": json.dumps({"transcript": transcript_to_analyze})}]),
                self.analyzer_agent.run([{"content": json.dumps({"transcript": transcript_to_analyze})}]),
                self.categorizer_agent.run([{"content": json.dumps({"transcript": transcript_to_analyze})}]),
                self.sarcasm_agent.run([{"content": json.dumps({"transcript": transcript_to_analyze})}])
            ]
            tox_result, analysis_result, categorization_result, sarcasm_result = await asyncio.gather(*tasks)

            logger.debug(f"[Orchestrator] ToxicityAgent → {tox_result}")
            logger.debug(f"[Orchestrator] AnalyzerAgent → {analysis_result}")
            logger.debug(f"[Orchestrator] CategorizerAgent → {categorization_result}")

            ctx.update(tox_result if isinstance(tox_result, dict) else {})
            ctx.update(analysis_result if isinstance(analysis_result, dict) else {})
            ctx.update(categorization_result if isinstance(categorization_result, dict) else {})
            ctx.update(sarcasm_result if isinstance(sarcasm_result, dict) else {})

            # ── decide which heavy agents we really need ─────────────────────────
            sentiment      = ctx.get("sentiment", "Neutral")
            tox_score      = ctx.get("toxicity", 0.0)          # 0-1 float
            sarcasm        = ctx.get("sarcasm", 0.0)
            need_abuse     = tox_score >= 0.60

            # 5. Caregiver scoring (empathy, responsiveness, engagement)
            scoring_result = await self.star_reviewer_agent.run(
                transcript_to_analyze,
                ctx.get("sentiment", "Neutral"),
                ctx.get("responsiveness", "Passive")
            )
            logger.debug(f"[Orchestrator] StarReviewerAgent → {scoring_result}")
            if isinstance(scoring_result, dict) and "error" in scoring_result:
                return scoring_result
            if isinstance(scoring_result, str):
                try:
                    scoring_result = json.loads(scoring_result)
                except json.JSONDecodeError:
                    return {"error": "Star reviewer output is not valid JSON."}
            ctx.update(scoring_result)

            # 7-10. Heavy analysis (run conditionally)
            tasks = [
                self.response_generator_agent.run([{"content": json.dumps(ctx)}]),
            ]
            if need_abuse:
                tasks.append(self.abuse_agent.run([{"content": json.dumps(ctx)}]))
            else:
                tasks.append(None)

            response_result, abuse_line = await asyncio.gather(
                *[t for t in tasks if t]            # filter out the Nones
            )

            logger.debug(f"[Orchestrator] ResponseGeneratorAgent → {response_result}")
            logger.debug(f"[Orchestrator] AbuseDetectionAgent → {abuse_line}")
            if response_result: ctx.update(response_result)
            if abuse_line: ctx.update(abuse_line)

            # 11. Global abuse flag
            ctx["abuse_flag"] = bool(ctx.get("abusive", False) or ctx.get("toxicity", False))
            return ctx

        except Exception as e:
            logger.exception("[Orchestrator] Unexpected error")
            return {"error": "Orchestrator failed: " + str(e)}