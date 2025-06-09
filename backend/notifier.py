# backend/notifier.py
"""
Writes a notification doc **and** pushes an FCM message.
No Cloud Functions, no Emulator required.
"""

import firebase_admin
from firebase_admin import credentials, firestore, messaging
from google.cloud.firestore_v1 import SERVER_TIMESTAMP

# ---------------------------------------------------------------------------
# 1) Firebase Admin SDK init (yalnızca 1 kez)
#    ENV değişkeni: GOOGLE_APPLICATION_CREDENTIALS = path/to/serviceAccount.json
# ---------------------------------------------------------------------------
if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.client()

# ---------------------------------------------------------------------------
def send_parent_notification(uid: str, ctx: dict) -> str:
    """
    Parameters
    ----------
    uid : str
        Parent user’s Firebase UID
    ctx : dict
        The analysis result coming from /analyze
    Returns
    -------
    str  Firestore doc ID (useful for logs)
    """

    # ---------- 2) Firestore: create notification doc ----------------------
    notif_ref = (
        db.collection("users")
          .document(uid)
          .collection("notifications")
          .document()        # auto-ID
    )

    notif_doc = {
        "title"   : ctx.get("title", "Care Interaction Alert"),
        "body"    : ctx.get("parent_notification", ""),
        "summary" : ctx.get("summary", ""),
        "timestamp": SERVER_TIMESTAMP,
        "read": False,

        "ctx_id":           ctx["id"],
        "primary_category": ctx["primary_category"],
        "category_group":   ctx["category_group"],
        "severity":         ctx.get("tone", 0),
        "abuse_flag":       bool(ctx.get("abuse_flag", False)),
        "sentiment":        ctx.get("sentiment", ""),
        "recommendations":  ctx.get("recommendations", []),
    }

    notif_ref.set(notif_doc)

    # ---------- 3) Grab all FCM tokens for this user -----------------------
    token_docs = (
        db.collection("users")
          .document(uid)
          .collection("device_tokens")
          .stream()
    )
    tokens = [d.id for d in token_docs]
    if not tokens:                                             # No devices
        return notif_ref.id

    # ---------- 4) Build & send push (HTTP v1, token başına send) ----------
    for tok in tokens:
        message = messaging.Message(
            token=tok,
            notification=messaging.Notification(
                title=notif_doc["title"],
                body =notif_doc["body"],
            ),
            data={
                "notifId": notif_ref.id,
                "ctxId"  : ctx["id"],
                "summary": notif_doc["summary"][:1024],
            },
        )

        # send()  →  FCM HTTP v1  (Kesinlikle /batch kullanmaz)
        messaging.send(message, dry_run=False)

    print(f"FCM v1: sent to {len(tokens)} token(s)")

    return notif_ref.id
