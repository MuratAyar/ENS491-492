"""Lightweight wrapper around your existing Orchestrator class.
   ‑ Finds the orchestrator automatically so you *don’t* have to touch imports.
"""
from __future__ import annotations

import importlib, asyncio, logging
from typing import Dict, Any

logger = logging.getLogger("ragos.analysis_pipeline")

# ---------------------------------------------------------------------------
#  Locate Orchestrator automatically
# ---------------------------------------------------------------------------
CANDIDATE_PATHS = [
    "agents.orchestration.orchestrator",  # as used in Streamlit app
    "agents.orchestrator",
    "orchestrator",  # root‑level file (uploaded example)
]

orchestrator = None
for mod_name in CANDIDATE_PATHS:
    try:
        mod = importlib.import_module(mod_name)
        orchestrator = mod.Orchestrator()
        logger.info("Loaded Orchestrator from %s", mod_name)
        break
    except (ImportError, AttributeError):
        continue
if orchestrator is None:  # pragma: no cover
    raise ImportError("Could not find an Orchestrator implementation – checked: " + ", ".join(CANDIDATE_PATHS))

# ---------------------------------------------------------------------------
#  Async helper – run pipeline
# ---------------------------------------------------------------------------
async def run_pipeline_async(transcript: str) -> Dict[str, Any]:
    """Ensures Orchestrator.process_transcript gets an event‑loop.
    Works whether that method is async or sync.
    """
    if asyncio.iscoroutinefunction(orchestrator.process_transcript):
        return await orchestrator.process_transcript(transcript)  # type: ignore[arg‑type]
    # fallback: run in thread
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, orchestrator.process_transcript, transcript)  # type: ignore[arg‑type]