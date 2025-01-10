from typing import Dict, Any
import json
from openai import OpenAI

class BaseAgent:
    def __init__(self, name: str, instructions: str):
        self.name = name
        self.instructions = instructions
        self.ollama_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        """Default run method to be overridden by child classes"""
        raise NotImplementedError("Subclass must implement run()")
    
    def _query_ollama(self, prompt: str) -> dict:
        """Query Ollama model with the given prompt"""
        try:
            print(f"Querying Ollama with prompt: {prompt}")
            response = self.ollama_client.chat.completions.create(
                model="llama3.1",
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            # Extract content and ensure it is JSON
            content = response.choices[0].message.content.strip()
            if not content.startswith("{"):  # Wrap in JSON if necessary
                content = json.dumps({"response": content})
            return json.loads(content)
        except Exception as e:
            print(f"Error querying Ollama: {str(e)}")
            return {"error": str(e)}



    
    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        """Safely parse JSON from text, handling potential errors"""
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                json_str = text[start : end + 1]
                return json.loads(json_str)
            return {"error": "No JSON content found"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON content"}
    
