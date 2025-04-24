from typing import Dict, Any
from agents.analyzer_agent import AnalyzerAgent
from agents.categorizer_agent import CategorizerAgent
from agents.response_generator_agent import ResponseGeneratorAgent
from agents.star_reviewer_agent import StarReviewerAgent
import json
import re
from datetime import datetime

class Orchestrator:
    def __init__(self):
        self.analyzer_agent = AnalyzerAgent()
        self.categorizer_agent = CategorizerAgent()
        self.response_generator_agent = ResponseGeneratorAgent()
        self.star_reviewer_agent = StarReviewerAgent()

    async def process_transcript(self, transcript: str) -> Dict[str, Any]:
        """Process a full caregiver-child conversation with timeline categorization."""
        print("[Orchestrator] Processing caregiver-child transcript")
        
        # Use the entire transcript as persistent history
        context = {"transcript": transcript}

        # Step 1: Analyze caregiver sentiment and responsiveness (BERT model)
        analyzer_result = await self.analyzer_agent.run([{"content": json.dumps(context)}])
        if "error" in analyzer_result:
            return {"error": f"Analyzer error: {analyzer_result['error']}"}
        context.update(analyzer_result)

        # Step 2: Categorize conversation topic (BERT model)
        categorizer_result = await self.categorizer_agent.run([{"content": json.dumps(context)}])
        if "error" in categorizer_result:
            return {"error": f"Categorizer error: {categorizer_result['error']}"}
        context.update(categorizer_result)

        # Step 3: Assign caregiver performance score (LLaMA 3.1)
        star_review_result = await self.star_reviewer_agent.run(
            transcript, 
            analyzer_result.get("sentiment", "Neutral"), 
            analyzer_result.get("responsiveness", "Passive")
        )
        if "error" in star_review_result:
            return {"error": f"Star reviewer error: {star_review_result['error']}"}
        context.update(star_review_result)

        # Step 4: Generate parent notification (LLaMA 3.1)
        response_result = await self.response_generator_agent.run([{"content": json.dumps(context)}])
        if "error" in response_result:
            return {"error": f"Response generator error: {response_result['error']}"}
        context.update(response_result)

        # Step 5: Generate timeline categorization output
        timeline_categories = []
        # Regex pattern: matches time brackets either with () or [] and captures the time and following text.
        pattern = r"[\(\[](\d{1,2}:\d{2}(?:\s?[AP]M)?)[]\)]\s*(.*)"
        matches = re.findall(pattern, context.get("transcript", ""))
        for time_str, segment in matches:
            result = self.categorizer_agent.classification_pipeline(segment, candidate_labels=self.categorizer_agent.categories)
            primary_category = result["labels"][0] if result and "labels" in result and result["labels"] else "Unknown"
            timeline_categories.append({"time": time_str, "category": primary_category})
        
        # Sort timeline categories by time (tries both 12-hour and 24-hour formats)
        def parse_time(t):
            for fmt in ("%I:%M %p", "%H:%M"):
                try:
                    return datetime.strptime(t, fmt)
                except ValueError:
                    continue
            return datetime.min

        timeline_categories.sort(key=lambda x: parse_time(x["time"]))
        context["timeline_categories"] = timeline_categories

        return context
