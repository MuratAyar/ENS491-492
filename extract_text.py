from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os

# File paths
PDF_PATH = "data/caregiver_best_practices.pdf"  # Ensure this file exists
INDEX_DIR = "embeddings/chroma_index"  # Not needed for `persist()`

def create_index():
    """Load caregiving guidelines, split text, and create a Chroma index."""
    print("Loading caregiver guidance PDF...")
    loader = PDFPlumberLoader(PDF_PATH)
    documents = loader.load()

    # Split the documents into chunks
    print("Splitting caregiver document into sections...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Explicitly specify LLM model for embeddings
    print("Creating embeddings for caregiver best practices...")
    embeddings = OllamaEmbeddings(model="openhermes:7b-mistral-v2.5-q5_1")  # Ensure this model is available
    vector_store = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=INDEX_DIR
    )
    vector_store.persist()

    # 🔹 Remove the incorrect persist line, since Chroma auto-persists now
    print("Chroma index created successfully!")

if __name__ == "__main__":
    create_index()