import json
import requests

class BaseAgent:
    def __init__(self, name: str, instructions: str):
        self.name = name
        self.instructions = instructions
        self.base_url = "http://localhost:11434/v1"
        self.api_key = "ollama"  # Or your actual API key

    def _query_ollama(self, prompt: str) -> dict:
        """Query Ollama model and extract valid JSON output."""
        try:
            print(f"[{self.name}] Querying Ollama with prompt: {prompt}")
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "llama3.2:3b-instruct-q4_K_S",
                "messages": [
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 256,
            }

            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                extracted_json = self._extract_json(content)
                return extracted_json
            else:
                print(f"[{self.name}] Error: {response.status_code} - {response.text}")
                return {"error": f"HTTP Error {response.status_code}"}

        except Exception as e:
            print(f"[{self.name}] Error querying Ollama: {str(e)}")
            return {"error": str(e)}

    def _extract_json(self, text: str) -> dict:
        """Extract valid JSON from a given text output by removing markdown artifacts."""
        try:
            if "```json" in text:
                start = text.find("```json") + len("```json")
                end = text.find("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                text = text[start:end].strip()

            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON from LLM response"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
