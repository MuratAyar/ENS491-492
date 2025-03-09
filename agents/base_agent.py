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

    async def run(self, messages: list):
        raise NotImplementedError("Subclass must implement run()")
    
    def _query_ollama(self, prompt: str) -> dict:
        """Query Ollama model and extract valid JSON output."""
        try:
            print(f"[{self.name}] Querying Ollama with prompt: {prompt}")
            response = self.ollama_client.chat.completions.create(
                model="llama3.1",
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            
            # Extract content and clean JSON
            content = response.choices[0].message.content.strip()
            extracted_json = self._extract_json(content)

            return extracted_json
        except Exception as e:
            print(f"[{self.name}] Error querying Ollama: {str(e)}")
            return {"error": str(e)}

    def _extract_json(self, text: str) -> dict:
        """Extracts valid JSON from a given text output by removing markdown artifacts."""
        try:
            # Remove Markdown-style JSON code blocks
            if "```json" in text:
                start = text.find("```json") + len("```json")
                end = text.find("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                text = text[start:end].strip()
            
            # Convert text to JSON
            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON from LLM response"}
