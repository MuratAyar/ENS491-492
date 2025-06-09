# backend/timeline_visibility.py
from typing import Dict

ANCHOR_GROUPS = {
    "Meals", "Sleep", "Hygiene", "Health", "Safety"
}
MILESTONES = {"First Word", "First Steps", "New Skill"}  # extend freely

def is_visible(ctx: Dict) -> bool:
    """Return True if this timeline card should be shown to parents."""
    if ctx.get("send_notification"):
        return True
    if ctx.get("abuse_flag"):
        return True
    if ctx.get("primary_category") in MILESTONES:
        return True
    group = ctx.get("category_group")
    if group in ANCHOR_GROUPS:
        return True
    # otherwise keep it hidden
    return False
