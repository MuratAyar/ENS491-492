import json
from agents.analysis.sarcasm_detection_agent import SarcasmDetectionAgent

import asyncio

agent = SarcasmDetectionAgent()
dummy = {"transcript": "[00:01] Caregiver: Well, *that* was brilliant…"}

async def main():
	res = await agent.run([{"content": json.dumps(dummy)}])
	print(res)

asyncio.run(main())
