"""
Shared Ollama client – every sub-agent inherits from here.
Default model is Qwen-7B, but can be overridden per agent.
"""

import json
import requests
from typing import Dict, Any
import logging
logger = logging.getLogger("care_monitor")     # global project logger



class BaseAgent:
    # ──────────────────────────────────────────────────────────────
    def __init__(self, name: str, instructions: str, model: str = "qwen:7b"):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.base_url = "http://localhost:11434/v1"  # Ollama endpoint

    # ──────────────────────────────────────────────────────────────
    def _query_ollama(self, prompt: str) -> Dict[str, Any]:
        """
        Send a prompt to the local Ollama server and try to return a JSON
        dictionary. If the model returns free-text instead of JSON, we fall
        back to returning it under the key ``raw_output`` so the UI can still
        display something rather than crash.
        """
        try:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.instructions},
                    {"role": "user",   "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 256,
            }

            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            logger.debug(f"[{self.name}] raw LLM output:\n{raw}\n---")
            return self._extract_json(raw)

        except Exception as e:
            # network/timeout/JSON issues – always return dict so callers are safe
            logger.exception("[Base Agent] crashed")
            return {"error": f"Ollama request failed: {e}"}

    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """
        Robust JSON extraction:

        • Removes ```json …``` or ``` fences if present.
        • Attempts to load the first {...} block.
        • If that fails, returns ``{"raw_output": <full_text>}``.
        """
        # strip code fences
        if "```" in text:
            # keep segments that are NOT the ```json fence tags
            parts = [
                seg for seg in text.split("```")
                if not seg.lower().strip().startswith("json")
            ]
            text = "\n".join(parts).strip()

        try:
            first = text.index("{")
            last = text.rindex("}") + 1
            return json.loads(text[first:last])
        except Exception:
            # couldn’t parse JSON – return raw string instead of crashing callers
            return {"raw_output": text.strip()}
