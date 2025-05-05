"""
Run quick toxicity check on a sample transcript and dump JSON result.

$ python toxicity_detection_test.py
"""
import json, asyncio, datetime, uuid
from pathlib import Path

from agents.analysis.toxicity_agent import ToxicityAgent

SAMPLE_TRANSCRIPT = """
[00:01] Child: Mom, I drew a dinosaur on the living-room wall!
[00:04] Caregiver: Oh, perfect—because that’s exactly where priceless art belongs, right?
[00:07] Child: Do you like it?
[00:09] Caregiver: Absolutely, the Louvre is going to beg to borrow our wall now.
[00:12] Child: Really?
[00:14] Caregiver: (laughs) Not quite, sweetheart—but the drawing is very creative.
[00:18] Child: Should I clean it?
[00:20] Caregiver: Let’s grab a damp cloth and make it a *portable* masterpiece instead.
"""

async def main():
    agent = ToxicityAgent()
    payload = {"transcript": SAMPLE_TRANSCRIPT}
    result = await agent.run([{"content": json.dumps(payload)}])
    print("ToxicityAgent result →", result)

    # ------------ save as JSON ------------------------------------------
    out_dir = Path("data/tests")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts  = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    uid = uuid.uuid4().hex[:6]
    file = out_dir / f"tox_test_{ts}_{uid}.json"
    with open(file, "w", encoding="utf-8") as f:
        json.dump({"input": payload, "output": result}, f, indent=2)
    print(f"Saved → {file}")

if __name__ == "__main__":
    asyncio.run(main())
