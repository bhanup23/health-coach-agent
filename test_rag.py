# test_rag.py
from rag.retriever import get_or_create_vector_store

if __name__ == "__main__":
    # Build or load the FAISS vector store
    store = get_or_create_vector_store()
    # Simple query to test retrieval
    query = "Can I drink coffee?"
    docs = store.similarity_search(query, k=3)
    print(f"Retrieved {len(docs)} documents for query: '{query}'")
    for i, doc in enumerate(docs, 1):
        print(f"Doc {i} preview: {doc.page_content[:100]}...")
