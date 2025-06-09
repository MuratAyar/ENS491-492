# backend/timeline.py
from __future__ import annotations
from datetime import datetime, timezone
from google.cloud import firestore_v1 as fs
from agents.analysis.category_utils import merge_window_of
from backend.timeline_visibility import is_visible   # NEW

try:
    from firebase.firebase_init import db  # type: ignore
except Exception as exc:
    raise ImportError("firebase/firebase_init.py must expose `db`: " + str(exc))

# ---------------------------------------------------------------------------
def _minutes(a: datetime, b: datetime) -> float:
    return abs((a - b).total_seconds()) / 60.0

def _should_merge(prev: dict, ctx: dict, now: datetime) -> bool:
    same_group  = prev["category_group"] == ctx["category_group"]
    window_mins = merge_window_of(ctx["category_group"])
    return same_group and _minutes(now, prev["end_time"]) <= window_mins
# ---------------------------------------------------------------------------
def update_timeline(*, user_id: str, ctx: dict,
                    result_id: str, ts_server: datetime) -> str | None:
    """
    Create/merge a timeline card **only if ctx is_visible**.
    Returns timeline-doc id or None if nothing was stored.
    """
    visible = is_visible(ctx)
    if not visible:
        return None                    # ←  early-exit: keep DB clean

    if ts_server.tzinfo is None:       # normalise
        ts_server = ts_server.replace(tzinfo=timezone.utc)

    tl_ref = (db.collection("users")
                .document(user_id)
                .collection("timeline"))

    last = next(iter(
        tl_ref.order_by("end_time", direction=fs.Query.DESCENDING)
              .limit(1).stream()), None)

    # ---------- merge -------------------------------------------------------
    if last and _should_merge(last.to_dict(), ctx, ts_server):
        doc_id  = last.id
        doc     = last.to_dict()
        n_old   = doc["metrics"]["count"]
        tl_ref.document(doc_id).update({
            "end_time"              : ts_server,
            "summary"               : ctx.get("summary", doc["summary"]),
            "metrics.avg_sentiment" : (doc["metrics"]["avg_sentiment"]*n_old
                                       + ctx["sentiment_score"]) / (n_old+1),
            "metrics.max_toxicity"  : max(doc["metrics"]["max_toxicity"],
                                          ctx["toxicity"]),
            "metrics.count"         : fs.Increment(1),
            "result_ids"            : fs.ArrayUnion([result_id]),
            "abuse_flag"            : doc["abuse_flag"] or ctx["abuse_flag"],
        })
        return doc_id

    # ---------- create new card --------------------------------------------
    new_doc = {
        "start_time"       : ts_server,
        "end_time"         : ts_server,
        "primary_category" : ctx["primary_category"],
        "category_group"   : ctx["category_group"],
        "snippet"          : ctx["transcript"].splitlines()[0][:120],
        "summary"          : ctx.get("summary", ""),
        "metrics": {
            "avg_sentiment": ctx["sentiment_score"],
            "max_toxicity" : ctx["toxicity"],
            "count"        : 1
        },
        "result_ids" : [result_id],
        "abuse_flag" : ctx["abuse_flag"],
    }
    ref = tl_ref.document()
    ref.set(new_doc)
    return ref.id
