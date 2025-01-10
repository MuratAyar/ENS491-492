from typing import Dict, Any
from agents.analyzer_agent import AnalyzerAgent
from agents.categorizer_agent import CategorizerAgent
from agents.response_generator_agent import ResponseGeneratorAgent
from agents.star_reviewer_agent import StarReviewerAgent  # Import the new agent
import pandas as pd
import json  # Add this import


class Orchestrator:
    def __init__(self):
        self.analyzer_agent = AnalyzerAgent()
        self.categorizer_agent = CategorizerAgent()
        self.response_generator_agent = ResponseGeneratorAgent()
        self.star_reviewer_agent = StarReviewerAgent()  # Initialize the StarReviewerAgent

    async def process_review(self, review: str) -> Dict[str, Any]:
        """Process a single review through all agents."""
        print("Orchestrator: Processing review")

        context = {"review": review}

        # Step 1: Analyze sentiment
        analyzer_result = await self.analyzer_agent.run([{"content": json.dumps(context)}])
        if "error" in analyzer_result:
            return {"error": f"Analyzer error: {analyzer_result['error']}"}

        sentiment = analyzer_result.get("analyzing_sentiment", "Neutral")
        context.update({"analyzing_sentiment": sentiment})

        # Step 2: Categorize the review
        categorizer_result = await self.categorizer_agent.run([{"content": json.dumps(context)}])
        if "error" in categorizer_result:
            return {"error": f"Categorizer error: {categorizer_result['error']}"}
        context.update({"category": categorizer_result.get("category", "General Feedback")})

        # Step 3: Generate response
        response_result = await self.response_generator_agent.run([{"content": json.dumps(context)}])
        if "error" in response_result:
            return {"error": f"Response generator error: {response_result['error']}"}
        context.update({"response": response_result.get("response", "No response generated.")})

        # Step 4: Predict expected stars
        star_review_result = await self.star_reviewer_agent.run(review, sentiment)
        if "error" in star_review_result:
            return {"error": f"Star reviewer error: {star_review_result['error']}"}
        context.update({"expected_stars": star_review_result.get("expected_stars", 3)})  # Default to 3 if missing


        return context

    async def process_reviews_csv(self, csv_path: str) -> pd.DataFrame:
        """Process reviews from a CSV file and return results as a DataFrame."""
        print("Orchestrator: Processing reviews from CSV")

        # Load reviews
        reviews_df = pd.read_csv(csv_path)
        if "translated_content" not in reviews_df.columns:
            raise ValueError("CSV file must contain a 'translated_content' column.")

        results = []
        for review in reviews_df["translated_content"]:
            if pd.isna(review) or not review.strip():
                results.append({"error": "Empty or missing review content"})
                continue

            # Process each review
            result = await self.process_review(review)
            results.append(result)

        # Add the "expected_stars" column
        results_df = pd.DataFrame(results)
        if "score" in reviews_df.columns:
            results_df["score"] = reviews_df["score"]

        return results_df
