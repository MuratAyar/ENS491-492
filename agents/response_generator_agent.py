from typing import Dict, Any
from .base_agent import BaseAgent
import json
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

class ResponseGeneratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ResponseGenerator",
            instructions="Generate only the response to the review based on sentiment and category. "
                         "Avoid any explanations, prefixes, or extra context. Provide clean, concise responses."
        )
        # Correct embedding function initialization with explicit model name
        self.vector_store = Chroma(
            persist_directory="embeddings/chroma_index",
            embedding_function=OllamaEmbeddings(model="llama3.1")  # Explicitly set model name
        )
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    async def run(self, messages: list) -> Dict[str, Any]:
        """Generate a response to the review"""
        try:
            review_data = json.loads(messages[-1]["content"])  # Parse JSON string
            review = review_data.get("review", "")
            sentiment = review_data.get("analyzing_sentiment", "Neutral")
            category = review_data.get("category", "General Feedback")

            if not review:
                return {"error": "No review content found to generate a response."}

            # Retrieve relevant context
            context = self.retriever.get_relevant_documents(review)
            context_text = "\n".join([doc.page_content for doc in context])

            # Create response generation prompt
            if sentiment.lower() == "positive":
                prompt = f"Thank the user for their positive feedback. Review: '{review}'"
            else:
                prompt = f"Provide a polite response based on the category '{category}'. Review: '{review}'"

            # Include context in the prompt
            prompt_with_context = f"Given the context:\n\n{context_text}\n\n{prompt}"

            response_result = self._query_ollama(prompt_with_context)

            # Clean response: remove unwanted prefixes or explanations
            raw_response = response_result.get("response", "").strip()
            cleaned_response = self._extract_clean_response(raw_response)

            # Ensure consistent format: add quotation marks if missing
            formatted_response = self._format_response(cleaned_response)

            return {"review": review, "response": formatted_response}
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            print(f"Error generating response: {e}")
            return {"error": "Failed to generate a response. Please check the input format."}

    def _extract_clean_response(self, text: str) -> str:
        """Extract the clean response text from the raw response."""
        # Remove common prefixes like "Here's a possible response:"
        unwanted_prefixes = [
            "Here’s a possible response:",
            "Here’s a potential response:",
            "Since the review is incomplete, I'll assume",
            "Here's an example of a polite response:"
        ]
        for prefix in unwanted_prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text

    def _format_response(self, text: str) -> str:
        """Ensure response is consistently formatted with or without quotes."""
        text = text.strip()
        # Add quotation marks if missing
        if not text.startswith('"') and not text.endswith('"'):
            return f'"{text}"'
        # Ensure single quotes are replaced with double quotes for consistency
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]  # Remove single quotes
            return f'"{text}"'
        return text
