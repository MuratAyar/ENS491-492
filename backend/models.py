"""Shared Pydantic data‑classes – expand as the API grows."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict, Any

class TranscriptIn(BaseModel):
    user_id: str = Field(..., min_length=1)
    transcript: str = Field(..., min_length=1)

class AnalysisDoc(BaseModel):
    id: str
    user_id: str
    timestamp: str
    # everything produced by your agents: dynamic -> Dict
    data: Dict[str, Any]