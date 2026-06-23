import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

APP_TITLE = "Health Coach AI Agent"

PROTOCOL_PDF_PATH = "data/protocol.pdf"
FAISS_INDEX_PATH = "faiss_index"

LLM_MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RETRIEVAL_TOP_K = 4


def get_llm():

    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables."
        )

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )


def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )