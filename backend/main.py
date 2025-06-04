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

import aiofiles
import json
from pathlib import Path
from google.cloud.firestore_v1 import DocumentSnapshot
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi import APIRouter
from fastapi import UploadFile
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
    
@app.post("/batch_analyze")
async def batch_analyze(
    request: Request,
    file: UploadFile,
):
    headers = dict(request.headers)
    client_key = headers.get("x-api-key")
    if client_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Read and parse JSON file
        async with aiofiles.open(file.file.fileno(), mode='r') as f:
            raw = await f.read()
        items = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")

    results = []
    for item in items:
        try:
            ctx = await run_pipeline_async(item["transcript"])
            doc_id = uuid.uuid4().hex

            firestore_data = {
                **ctx,
                "id": doc_id,
                "user_id": item["user_id"],
                "timestamp": SERVER_TIMESTAMP
            }

            db.collection("users")\
              .document(item["user_id"])\
              .collection("analysis_results")\
              .document(doc_id).set(firestore_data)

            ctx.update({
                "id": doc_id,
                "user_id": item["user_id"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            results.append({"status": "success", "data": ctx})
        except Exception as e:
            logger.error("Error analyzing entry: %s", e)
            results.append({"status": "error", "user_id": item.get("user_id"), "detail": str(e)})

    return {"summary": {"success": sum(r["status"] == "success" for r in results),
                        "error": sum(r["status"] == "error" for r in results)},
            "results": results}

def clean_firestore_doc(doc: dict) -> dict:
    """Datetime nesnelerini string'e çevirir."""
    cleaned = {}
    for k, v in doc.items():
        if isinstance(v, datetime):
            cleaned[k] = v.isoformat()
        else:
            cleaned[k] = v
    return cleaned

@app.get("/export_all_analysis")
async def export_all_analysis(request: Request):
    headers = dict(request.headers)
    client_key = headers.get("x-api-key")
    if client_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        users_ref = db.collection("users")
        users = users_ref.stream()

        all_results = []

        for user_doc in users:
            user_id = user_doc.id
            results_ref = users_ref.document(user_id).collection("analysis_results")
            result_docs = results_ref.stream()

            for doc in result_docs:
                raw = doc.to_dict()
                raw = clean_firestore_doc(raw)  # 🧼 tarihleri düzelt
                raw["user_id"] = user_id
                raw["doc_id"] = doc.id
                all_results.append(raw)

        output_path = Path(__file__).resolve().parent / "all_analysis_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        return FileResponse(
            path=str(output_path),
            filename="all_analysis_results.json",
            media_type="application/json"
        )

    except Exception as ex:
        logger.exception("Export failed")
        raise HTTPException(500, detail=str(ex))