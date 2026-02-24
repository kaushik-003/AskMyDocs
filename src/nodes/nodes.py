"""LangGraph Nodes for RAG Workflow."""

from src.state.rag_state import RAGState


class RAGNodes:
    """Node functions for RAG workflow."""

    def __init__(self, retriever, llm):
        """Initializes the RAGNodes with a retriever and LLM.

        Args:
            retriever: The retriever to use for fetching relevant documents.
            llm: The language model instance.
        """
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents based on the question in the state.

        Args:
            state: The current RAGState.
        Returns:
            Updated RAGState with retrieved documents.
        """
        retrieved_docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=retrieved_docs,
        )

    def generate_response(self, state: RAGState) -> RAGState:
        """Generate a response based on the question and retrieved documents.

        Args:
            state: The current RAGState.
        Returns:
            Updated RAGState with generated answer.
        """
        context = "\n\n".join(doc.page_content for doc in state.retrieved_docs)

        prompt = (
            f"Answer the question based on the context provided.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state.question}"
        )

        response = self.llm.invoke(prompt)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content,
        )