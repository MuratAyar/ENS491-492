import asyncio
import json
import sys
from agents.orchestrator import Orchestrator

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <transcript_text_file>")
        sys.exit(1)
    transcript_file = sys.argv[1]
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_text = f.read().strip()
    except Exception as e:
        print(f"Failed to read file '{transcript_file}': {e}")
        sys.exit(1)
    if not transcript_text:
        print("Transcript file is empty.")
        sys.exit(1)
    orchestrator = Orchestrator()
    result = asyncio.run(orchestrator.process_transcript(transcript_text))
    # Print the result as formatted JSON
    print(json.dumps(result, ensure_ascii=False, indent=2))
    