"""LangGraph Nodes for RAG Workflow."""

from src.state.rag_state import RAGState


class RAGNodes:
    """node functions for RAG workflow."""
    def __init__(self, retriever, llm):
        """Initializes the RAGNodes with a retriever and LLM.
        
        Args:
            retriever: The retriever to use for fetching relevant documents.
            llm: The language model instance
        """
        self.retriever = retriever
        self.llm = llm

        def retrieve_docs(self, state: RAGState) -> RAGState:
            """Node function to retrieve relevant documents based on the question in the state.
            
            Args:
                state: The current RAGState.
            Returns:
                Updated RAGState with retrieved documents.
            """
            retrieved_docs = self.retriever.retrieve(state.question)
            return RAGState(
                question=state.question,
                retrieved_docs=retrieved_docs
            )
        
        def generate_response(self, state: RAGState) -> RAGState:
            """Node function to generate a response based on the question and retrieved documents in the state.
            
            Args:
                state: The current RAGState.
            Returns:
                Updated RAGState with generated answer.
            """
            context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
            
            prompt = f"""Answer the question based on the context provided. 

            Context:{context}

            Question: {state.question}
            """

            response = self.llm.generate(prompt)

            return RAGState(
                question=state.question,
                retrieved_docs=state.retrieved_docs,
                answer=response.content
            )