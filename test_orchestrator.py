import asyncio
import os
from agents.orchestrator import Orchestrator

async def test_csv():
    orchestrator = Orchestrator()
    
    # Dynamically resolve the path to the CSV file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "yk_translated_content_copy.csv")

    print(f"Testing reviews from CSV at: {csv_path}")
    
    try:
        results_df = await orchestrator.process_reviews_csv(csv_path)

        # Ensure output directory exists
        output_dir = os.path.join(base_dir, "data")
        os.makedirs(output_dir, exist_ok=True)

        # Save the results
        results_path = os.path.join(output_dir, "results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
    except Exception as e:
        print(f"Error during test: {e}")

# Run the test
asyncio.run(test_csv())
