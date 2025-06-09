"""FastAPI entry-point for the RAGOS Care-Monitor backend."""
from __future__ import annotations

import os, uuid, logging, asyncio, json
from typing import Dict, Any
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ─── Google Firestore --------------------------------------------------------
from google.cloud import firestore_v1 as fs            # NEW alias
from google.cloud.firestore_v1 import SERVER_TIMESTAMP

# ─── Local modules -----------------------------------------------------------
from backend.timeline import update_timeline
from backend.aggregator import compute_aggregates
from backend.analysis_pipeline import orchestrator, run_pipeline_async  # adjust import if path differs

from backend.notifier import send_parent_notification

# -----------------------------------------------------------------------------
#  ENV & Logging
# -----------------------------------------------------------------------------
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("RAGOS_API_KEY")
if not API_KEY:
    logging.warning("RAGOS_API_KEY env var is empty!")

logger = logging.getLogger("ragos.backend")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# -----------------------------------------------------------------------------
#  Firestore init  (expects firebase_init.py to expose `db`)
# -----------------------------------------------------------------------------
try:
    from firebase.firebase_init import db  # type: ignore
except Exception as e:
    raise ImportError("firebase/firebase_init.py must expose Firestore `db`: " + str(e))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "analysis_results")

# -----------------------------------------------------------------------------
#  FastAPI setup
# -----------------------------------------------------------------------------
app = FastAPI(title="RAGOS-API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
#  Pydantic models
# -----------------------------------------------------------------------------
class TranscriptIn(BaseModel):
    user_id: str = Field(..., example="user_123")
    transcript: str = Field(..., example="[00:01] Child: ...")

class AnalysisOut(BaseModel):
    status: str = "success"
    data: Dict[str, Any]

# -----------------------------------------------------------------------------
#  Routes
# -----------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "server_time": datetime.now(timezone.utc).isoformat()}

# ------------------------------------------------------------------ /analyze
@app.post("/analyze", response_model=AnalysisOut)
async def analyze(payload: TranscriptIn, request: Request):
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        ctx: Dict[str, Any] = await run_pipeline_async(payload.transcript)
    except Exception as ex:
        logger.exception("Agent pipeline crashed")
        raise HTTPException(500, detail=str(ex))

    doc_id = uuid.uuid4().hex

    firestore_data = {
        **ctx,
        "id": doc_id,
        "user_id": payload.user_id,
        "timestamp": SERVER_TIMESTAMP
    }

    try:
        (db.collection("users")
           .document(payload.user_id)
           .collection("analysis_results")
           .document(doc_id)
           .set(firestore_data))

        # ---- timeline merge -------------
        update_timeline(
            user_id   = payload.user_id,
            ctx       = ctx,
            result_id = doc_id,
            ts_server = datetime.now(timezone.utc)
        )

        # ---- push-notification kaydı ----
        if ctx.get("send_notification"):
            send_parent_notification(payload.user_id, {**ctx, "id": doc_id})

    except Exception as ex:
        logger.error("Firestore write failed: %s", ex)
        raise HTTPException(500, detail=str(ex))

    # Return API-friendly timestamp
    ctx.update({
        "id": doc_id,
        "user_id": payload.user_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    return {"status": "success", "data": ctx}

# ---------------------------------------------------------- aggregates route
@app.get("/aggregate/{user_id}")
async def get_aggregates(user_id: str):
    try:
        return {"status": "success", "data": compute_aggregates(user_id)}
    except Exception as ex:
        logger.exception("Aggregation failed")
        raise HTTPException(500, detail=str(ex))

# ----------------------------------------------------------- timeline route
@app.get("/timeline/{user_id}")
async def get_timeline(user_id: str, day: str | None = None, limit: int = 50):
    col = (db.collection("users")
             .document(user_id)
             .collection("timeline"))
    if day:
        from dateutil import tz
        local = datetime.fromisoformat(day).replace(tzinfo=tz.gettz())         # 🟡 treat as local
        start = local.astimezone(timezone.utc)                                 # convert to UTC
        end   = (local + timedelta(days=1)).astimezone(timezone.utc)

        q = (col.where("start_time", ">=", start)
                 .where("start_time", "<",  end)
                 .order_by("start_time"))
    else:
        q = col.order_by("start_time", direction=fs.Query.DESCENDING).limit(limit)

    docs = [{**d.to_dict(), "id": d.id} for d in q.stream()]
    for d in docs:
        for k in ("start_time", "end_time"):
            if isinstance(d.get(k), datetime):
                d[k] = d[k].isoformat()
    return {"status": "success", "data": docs}

# --------------------------------------------------------------------------- 
#  (Other routes like /batch_analyze, /export_all_analysis remain unchanged.)
