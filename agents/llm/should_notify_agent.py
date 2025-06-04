# agents/llm/should_notify_agent.py
import json, logging
from typing import Dict, Any, List
from .base_agent import BaseAgent

logger = logging.getLogger("care_monitor")

class ShouldNotifyAgent(BaseAgent):
    """
    Returns { "notify": true/false, "reason": "..."} via LLM judging.
    """

    def __init__(self):
        super().__init__(
            name="ShouldNotify",
            instructions=(
                """You decide whether a caregiver–child interaction requires
        a push-notification for parents.
        Return STRICT JSON only:
        { "notify": true,  "reason": "string" }
        or
        { "notify": false, "reason": "string" }

        Notify when:
        • Potential harm, yelling, shaming, or threats.
        • Repetition of unhealthy patterns.
        • Developmental milestones parents would value.
        If unsure, default to false."""
            )
        )


    async def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        ### METRICS
        sentiment_score : {ctx.get('sentiment_score')}
        toxicity        : {ctx.get('toxicity')}
        caregiver_score : {ctx.get('caregiver_score')}
        tone            : {ctx.get('tone')}
        empathy         : {ctx.get('empathy')}
        responsiveness  : {ctx.get('responsiveness')}
        sarcasm         : {ctx.get('sarcasm')}
        abuse_flag      : {ctx.get('abuse_flag')}
        primary_category: {ctx.get('primary_category')}
        secondary_cat[] : {ctx.get('secondary_categories')}
        ### CONVERSATION
        {ctx.get('transcript')[:1200]}
        ### STRICT OUTPUT JSON
        {{ "notify": false, "reason": "" }}
        """
        out = self._query_ollama(prompt)

        # Güvenle JSON çek
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except Exception:
                pass
        if isinstance(out, dict) and "notify" in out:
            return out
        return {"notify": False, "reason": "parse error"}
