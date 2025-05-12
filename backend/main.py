"""FastAPI entry‑point for the RAGOS Care‑Monitor backend.

 ▸ POST  /analyze   →  run LLM agents on a transcript, persist to Firestore
 ▸ GET   /health    →  simple health‑check

Env vars used
-------------
FIREBASE_CREDENTIALS  – path to serviceAccountKey.json (optional if firebase_init does this)
ORCHESTRATOR_PATH     – dotted path to Orchestrator (fallbacks are tried automatically)
COLLECTION_NAME       – Firestore collection (default: analysis_results)
"""
from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

import os, sys, uuid, logging, asyncio
from datetime import datetime, timezone
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter
from pydantic import BaseModel, Field
import logging
logging.basicConfig(level=logging.DEBUG)

from backend.aggregator import compute_aggregates


API_KEY = os.getenv("RAGOS_API_KEY")
print(f"ENV API_KEY from dotenv = {API_KEY}")  # Log olarak bas
if not API_KEY:
    logging.warning("RAGOS_API_KEY env var is empty!")

# ---- Internal imports -----------------------------------------------------
from .analysis_pipeline import orchestrator, run_pipeline_async

try:
    # your firebase_init.py must expose `db` (firestore client)
    from firebase.firebase_init import db  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("firebase/firebase_init.py must expose a Firestore client named `db` – " + str(e))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "analysis_results")
logger = logging.getLogger("ragos.backend")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

app = FastAPI(title="RAGOS‑API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in prod!!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
#  Pydantic models
# ---------------------------------------------------------------------------
class TranscriptIn(BaseModel):
    """Input payload coming from Flutter or any client."""

    user_id: str = Field(..., example="user_123")
    transcript: str = Field(..., example="[00:01] Child: ...")

class AnalysisOut(BaseModel):
    """API response for /analyze."""

    status: str = "success"
    data: Dict[str, Any]


# ---------------------------------------------------------------------------
#  Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "server_time": datetime.utcnow().isoformat()}


@app.post("/analyze", response_model=AnalysisOut)
async def analyze(
    payload: TranscriptIn,
    request: Request
):
    headers = dict(request.headers)
    client_key = headers.get("x-api-key")

    if client_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        ctx: Dict[str, Any] = await run_pipeline_async(payload.transcript)
    except Exception as ex:
        logger.exception("Agent pipeline crashed")
        raise HTTPException(500, detail=str(ex))

    doc_id = uuid.uuid4().hex

    # Firestore için ayrı veri (timestamp = sunucu zamanı)
    firestore_data = {
        **ctx,
        "id": doc_id,
        "user_id": payload.user_id,
        "timestamp": SERVER_TIMESTAMP
    }

    try:
        doc_ref = (
            db.collection("users")
              .document(payload.user_id)
              .collection("analysis_results")
              .document(doc_id)
        )
        doc_ref.set(firestore_data)
    except Exception as ex:
        logger.error("Failed to write to Firestore: %s", ex)
        raise HTTPException(500, detail="Firestore write failed: " + str(ex))

    # API cevabına UTC saatli string timestamp ekle
    ctx.update({
        "id": doc_id,
        "user_id": payload.user_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return {"status": "success", "data": ctx}

@app.get("/aggregate/{user_id}")
async def get_aggregates(user_id: str):
    """Kullanıcının saatlik / günlük / haftalık skor ortalamalarını döner."""
    try:
        data = compute_aggregates(user_id)
        return {"status": "success", "data": data}
    except Exception as ex:
        logger.exception("Aggregation failed")
        raise HTTPException(500, detail=str(ex))
