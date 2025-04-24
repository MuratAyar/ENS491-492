from typing import Dict, Any
from .base_agent import BaseAgent
import json

# Updated imports from the community packages:
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Import the persistent Chroma client from chromadb
from chromadb import PersistentClient
from chromadb.config import Settings

class ResponseGeneratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ParentNotifier",
            instructions=(
                "Generate a caregiver interaction summary for parents. "
                "Highlight positive caregiving aspects and potential concerns. "
                "Provide clear, structured feedback in JSON format."
            )
        )

        # Initialize the persistent Chroma client with new settings:
        # Using a folder "embeddings/chroma_index" to store data persistently.
        # The `allow_reset` flag enables the reset() method.
        self.vector_store = PersistentClient(
            path="embeddings/chroma_index",
            settings=Settings(
                allow_reset=True,
            )
        )

        # Initialize the embedding function from langchain_community
        embedding_function = OllamaEmbeddings(model="llama3.1")

        # Create the Chroma-based retriever using the persistent vector store
        self.retriever = Chroma(
            client=self.vector_store,
            embedding_function=embedding_function
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        """Generate a summary notification for parents."""
        print("[ParentNotifier] Generating caregiver performance summary")
        try:
            transcript_data = json.loads(messages[-1]["content"])
            transcript = transcript_data.get("transcript", "")
            sentiment = transcript_data.get("sentiment", "Neutral")
            category = transcript_data.get("primary_category", "General")
            caregiver_score = transcript_data.get("caregiver_score", 3)
            feedback = transcript_data.get("feedback", "")

            if not transcript:
                return {"error": "No transcript content found for generating a response."}

            # Retrieve caregiving best practices related to category
            context_docs = self.retriever.similarity_search(category, k=2)
            context_text = "\n".join([doc.page_content for doc in context_docs])

            # Create parent notification prompt
            prompt = (
                f"Generate a caregiver evaluation summary for parents based on this conversation:\n\n{transcript}\n\n"
                f"Caregiver performance metrics:\n"
                f"- Sentiment: {sentiment}\n"
                f"- Category: {category}\n"
                f"- Caregiver Score: {caregiver_score}/5\n\n"
                f"Feedback:\n{feedback}\n\n"
                f"Use caregiving best practices from:\n{context_text}\n\n"
                "Provide a summary in JSON format as:\n"
                '{ "parent_notification": "<brief, polite summary>", '
                '"recommendations": ["suggestions for improvement, if needed"] }'
            )

            # Query the Ollama model (using inherited _query_ollama method)
            response_result = self._query_ollama(prompt)

            # Ensure JSON format is extracted before returning
            if isinstance(response_result, str):
                response_result = self._extract_json(response_result)

            # Fix recommendations if returned as a simple list
            if isinstance(response_result.get("recommendations"), list):
                response_result["recommendations"] = [
                    {"category": "General", "description": rec} if isinstance(rec, str) else rec
                    for rec in response_result["recommendations"]
                ]

            print(f"[ParentNotifier] Parent notification result: {response_result}")

            if "error" in response_result:
                return {"error": response_result["error"]}

            return response_result

        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            print(f"[ParentNotifier] Error generating response: {e}")
            return {"error": "Failed to generate parent notification."}
