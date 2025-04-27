from typing import Dict, Any
import json, re, logging, asyncio
from datetime import datetime

# Import agents
from agents.language_switch_agent import LanguageSwitchAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.categorizer_agent import CategorizerAgent
from agents.response_generator_agent import ResponseGeneratorAgent
from agents.star_reviewer_agent import StarReviewerAgent
from agents.toxicity_agent import ToxicityAgent
from agents.timeline_sentiment_agent import TimelineSentimentAgent
from agents.emotion_analysis_agent import EmotionAnalysisAgent
from agents.abuse_detection_agent import AbuseDetectionAgent

# Setup logger
logger = logging.getLogger("care_monitor")

class Orchestrator:
    """Runs all sub-agents in sequence and returns a combined result."""
    
    def __init__(self):
        self.language_agent = LanguageSwitchAgent(target_language="en")
        self.analyzer_agent = AnalyzerAgent()
        self.categorizer_agent = CategorizerAgent()
        self.star_reviewer_agent = StarReviewerAgent()
        self.response_generator_agent = ResponseGeneratorAgent()
        self.tox_agent = ToxicityAgent()
        self.tline_agent = TimelineSentimentAgent()
        self.emotion_agent = EmotionAnalysisAgent()
        self.abuse_agent = AbuseDetectionAgent()

    async def process_transcript(self, transcript: str) -> Dict[str, Any]:
        """Process the input transcript through all analysis agents."""
        ctx: Dict[str, Any] = {}
        try:
            # 1. Language detection and optional translation
            lang_result = await self.language_agent.run(transcript)
            logger.debug(f"[Orchestrator] LanguageSwitchAgent → {lang_result}")
            if "error" in lang_result:
                print(f"[Orchestrator] Language detection error: {lang_result['error']}")
            ctx.update({"transcript": transcript})
            ctx.update(lang_result)
            transcript_to_analyze = ctx.get("transcript", transcript)

            # 2-4. Parallel quick analysis: toxicity, sentiment & tone, topic categorization
            tasks = [
                self.tox_agent.run([{"content": json.dumps({"transcript": transcript_to_analyze})}]),
                self.analyzer_agent.run([{"content": json.dumps({"transcript": transcript_to_analyze})}]),
                self.categorizer_agent.run([{"content": json.dumps({"transcript": transcript_to_analyze})}])
            ]
            tox_result, analysis_result, categorization_result = await asyncio.gather(*tasks)
            logger.debug(f"[Orchestrator] ToxicityAgent → {tox_result}")
            logger.debug(f"[Orchestrator] AnalyzerAgent → {analysis_result}")
            logger.debug(f"[Orchestrator] CategorizerAgent → {categorization_result}")
            ctx.update(tox_result if isinstance(tox_result, dict) else {})
            ctx.update(analysis_result if isinstance(analysis_result, dict) else {})
            ctx.update(categorization_result if isinstance(categorization_result, dict) else {})

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

            # 6. Timeline category classification (per segment topics)
            timeline = []
            segments = re.findall(r"[\(\[](\d{1,2}:\d{2}(?:\s?[AP]M)?)[]\)]\s*(.*)", transcript)
            if segments:
                times = [t for t, seg in segments]
                texts = [seg[:512] for _, seg in segments]  # limit length per segment for efficiency
                try:
                    results = self.categorizer_agent.classification_pipeline(texts, candidate_labels=self.categorizer_agent.categories)
                except Exception as e:
                    results = []
                    for text in texts:
                        try:
                            out = self.categorizer_agent.classification_pipeline(text, candidate_labels=self.categorizer_agent.categories)
                        except Exception:
                            out = {}
                        results.append(out)
                # Ensure results is list of outputs
                if isinstance(results, dict):
                    results_list = [results]
                else:
                    results_list = results or []
                for t, res in zip(times, results_list):
                    label = res.get("labels", ["Unknown"])[0] if res else "Unknown"
                    timeline.append({"time": t, "category": label})
            def _parse_time(ts: str):
                for fmt in ("%I:%M %p", "%H:%M"):
                    try:
                        return datetime.strptime(ts, fmt)
                    except ValueError:
                        continue
                return datetime.min
            ctx["timeline_categories"] = sorted(timeline, key=lambda x: _parse_time(x["time"]))

            # 7-10. Parallel heavy analysis: parent response, timeline sentiment, emotions, abuse
            tasks = [
                self.response_generator_agent.run([{"content": json.dumps(ctx)}]),
                self.tline_agent.run([{"content": json.dumps(ctx)}]),
                self.emotion_agent.run([{"content": json.dumps(ctx)}]),
                self.abuse_agent.run([{"content": json.dumps(ctx)}])
            ]
            response_result, sent_line, emotion_line, abuse_line = await asyncio.gather(*tasks)
            logger.debug(f"[Orchestrator] ResponseGeneratorAgent → {response_result}")
            logger.debug(f"[Orchestrator] TimelineSentimentAgent → {sent_line}")
            logger.debug(f"[Orchestrator] EmotionAnalysisAgent → {emotion_line}")
            logger.debug(f"[Orchestrator] AbuseDetectionAgent → {abuse_line}")
            if response_result: ctx.update(response_result)
            if sent_line: ctx.update(sent_line)
            if emotion_line: ctx.update(emotion_line)
            if abuse_line: ctx.update(abuse_line)

            # 11. Global abuse flag
            ctx["abuse_flag"] = bool(ctx.get("abusive", False) or ctx.get("toxicity", False))
            return ctx

        except Exception as e:
            logger.exception("[Orchestrator] Unexpected error")
            return {"error": "Orchestrator failed: " + str(e)}
