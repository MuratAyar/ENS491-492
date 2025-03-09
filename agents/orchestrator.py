from typing import Dict, Any
from agents.analyzer_agent import AnalyzerAgent
from agents.categorizer_agent import CategorizerAgent
from agents.response_generator_agent import ResponseGeneratorAgent
from agents.star_reviewer_agent import StarReviewerAgent
import json

class Orchestrator:
    def __init__(self):
        self.analyzer_agent = AnalyzerAgent()
        self.categorizer_agent = CategorizerAgent()
        self.response_generator_agent = ResponseGeneratorAgent()
        self.star_reviewer_agent = StarReviewerAgent()

    async def process_transcript(self, transcript: str) -> Dict[str, Any]:
        """Process a full caregiver-child conversation"""
        print("[Orchestrator] Processing caregiver-child transcript")

        context = {"transcript": transcript}

        # Step 1: Analyze caregiver sentiment and responsiveness
        analyzer_result = await self.analyzer_agent.run([{"content": json.dumps(context)}])
        if "error" in analyzer_result:
            return {"error": f"Analyzer error: {analyzer_result['error']}"}

        context.update(analyzer_result)

        # Step 2: Categorize conversation topic
        categorizer_result = await self.categorizer_agent.run([{"content": json.dumps(context)}])
        if "error" in categorizer_result:
            return {"error": f"Categorizer error: {categorizer_result['error']}"}
        
        context.update(categorizer_result)

        # Step 3: Assign caregiver performance score
        star_review_result = await self.star_reviewer_agent.run(
            transcript, 
            analyzer_result.get("sentiment", "Neutral"), 
            analyzer_result.get("responsiveness", "Passive")
        )
        if "error" in star_review_result:
            return {"error": f"Star reviewer error: {star_review_result['error']}"}

        context.update(star_review_result)

        # Step 4: Generate parent notification
        response_result = await self.response_generator_agent.run([{"content": json.dumps(context)}])
        if "error" in response_result:
            return {"error": f"Response generator error: {response_result['error']}"}

        context.update(response_result)

        return context
