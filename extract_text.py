from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os

# File paths
PDF_PATH = "data/responding-to-customer-reviews.pdf"
INDEX_DIR = "embeddings/chroma_index"

def create_index():
    # Load the PDF document
    print("Loading PDF...")
    loader = PDFPlumberLoader(PDF_PATH)
    documents = loader.load()

    # Split the documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Generate embeddings and create Chroma index
    print("Creating embeddings and Chroma index...")
    embeddings = OllamaEmbeddings()
    vector_store = Chroma.from_documents(texts, embeddings)
    vector_store.persist(INDEX_DIR)
    print(f"Chroma index saved to {INDEX_DIR}")

if __name__ == "__main__":
    create_index()
