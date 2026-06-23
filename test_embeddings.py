# test_embeddings.py
from utils.config import get_embeddings

if __name__ == "__main__":
    embeddings = get_embeddings()
    try:
        vector = embeddings.embed_query("This is a test sentence for embeddings.")
        print("SUCCESS: Retrieved vector of length", len(vector))
    except Exception as e:
        print("Embeddings error:", e)
