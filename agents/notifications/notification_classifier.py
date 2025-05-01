# agents/notifications/notification_classifier.py
from typing import Dict, Any


class NotificationClassifier:
    """
    Determines if a given analysis result should be sent as a notification to the parent.
    This is a rule-based system that can later be upgraded to a learned classifier.
    """
    def should_notify(self, ctx: Dict[str, Any]) -> bool:
        # 1. High toxicity
        if ctx.get("toxicity", 0) > 0.7:
            return True

        # 2. Negative sentiment + low responsiveness
        if ctx.get("sentiment") == "Negative" and ctx.get("responsiveness") == "Passive":
            return True

        # 3. Certain high-priority categories
        if ctx.get("primary_category", "").lower() in ["emotional support", "safety", "physical abuse"]:
            return True

        # 4. Low caregiver performance score
        if float(ctx.get("caregiver_score", 10)) <= 4.0:
            return True

        # 5. Explicit abuse flag
        if ctx.get("abuse_flag", False):
            return True

        # Default: don't notify
        return False
