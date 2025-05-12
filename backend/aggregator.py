# backend/aggregator.py  (tam dosya)

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
from firebase.firebase_init import db
from google.cloud.firestore_v1 import DocumentSnapshot

NUMERIC_KEYS = {
    "sentiment_score", "toxicity", "sarcasm",
    "caregiver_score", "tone", "empathy", "responsiveness",
}

def _collect_since(user_id: str, since: datetime) -> List[Dict[str, Any]]:
    coll = (db.collection("users")
              .document(user_id)
              .collection("analysis_results"))
    # Eğer Firestore’ da string timestamp’ler de varsa ikili sorgu mümkün değil.
    # Çözüm: önce Timestamp tipinde sorgula, sonra string’leri filtrele.
    qs = coll.where("timestamp", ">=", since)
    docs = [d.to_dict() for d in qs.stream()]

    # Ek olarak elde kalan string timestamp kayıtlarını manuel filtrele
    iso_docs = [d.to_dict() for d in coll.where("timestamp", "==", None).stream()]  # None olup olmadığına bakma hilesi
    for d in iso_docs:
        try:
            ts = datetime.fromisoformat(d["timestamp"]).replace(tzinfo=timezone.utc)
            if ts >= since:
                docs.append(d)
        except Exception:
            pass
    return docs

def _mean(v: List[float]) -> float: return sum(v) / len(v) if v else 0.0
def _round(x: float) -> float:      return round(x, 1)
def _label(v: float) -> str:
    return "positive" if v > 0.2 else "negative" if v < -0.2 else "neutral"

def compute_aggregates(user_id: str) -> Dict[str, Any]:
    now, week = datetime.now(timezone.utc), timedelta(days=7)
    docs = _collect_since(user_id, now - week)

    hourly_cut = now - timedelta(hours=1)
    daily_cut  = now - timedelta(days=1)
    windows    = {"hourly": [], "daily": [], "weekly": docs}

    for d in docs:
        ts = d["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts).astimezone(timezone.utc)
        elif isinstance(ts, datetime):
            ts = ts.astimezone(timezone.utc)
        else:
            raise ValueError(f"Unknown timestamp format: {type(ts)}")

        if ts >= hourly_cut: windows["hourly"].append(d)
        if ts >= daily_cut:  windows["daily"].append(d)

    out: Dict[str, Any] = {}
    for win, rows in windows.items():
        agg: Dict[str, float] = {
            k: _round(_mean([r.get(k, 0.0) for r in rows if isinstance(r.get(k), (int, float))]))
            for k in NUMERIC_KEYS
        }
        agg["count"]           = len(rows)
        agg["sentiment_label"] = _label(agg["sentiment_score"])
        out[win] = agg
    return out
