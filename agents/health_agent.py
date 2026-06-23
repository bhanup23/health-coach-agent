from rag.retriever import get_or_create_vector_store


class HealthAgent:

    def __init__(self):

        self.vector_store = (
            get_or_create_vector_store()
        )

    def generate_response(
        self,
        user_message,
        chat_history=None
    ):

        docs = self.vector_store.similarity_search(
            user_message,
            k=4
        )

        context = "\n\n".join(
            [
                doc.page_content
                for doc in docs
            ]
        )

        if len(docs) == 0:
            answer = (
                "I cannot find that information "
                "in the wellness protocol."
            )
        else:
            answer = (
                "Based on the wellness protocol:\n\n"
                + context[:1000]
            )

        return answer, docs