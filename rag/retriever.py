# rag/retriever.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config import FAISS_INDEX_PATH, PROTOCOL_PDF_PATH, get_embeddings

def load_documents():
    loader = PyPDFLoader(PROTOCOL_PDF_PATH)
    documents = loader.load()
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(documents)

def build_vector_store():
    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store

def load_vector_store():
    embeddings = get_embeddings()
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def get_or_create_vector_store():
    """
    Load existing FAISS index if it exists; otherwise, build a new one.
    """
    index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    if os.path.exists(index_file):
        return load_vector_store()
    return build_vector_store()

def retrieve_context(query: str, k: int = 4):
    """
    Retrieve the top-k matching documents for the query.
    """
    vector_store = get_or_create_vector_store()
    docs = vector_store.similarity_search(query, k=k)
    return docs
