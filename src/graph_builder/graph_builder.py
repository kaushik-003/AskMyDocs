"""Graph builder for LangGraph Workflow."""

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.nodes import RAGNodes

class GraphBuilder:
    """Builds and Manages the RAG workflow graph."""
    
    def __init__(self,retriever, llm):
        """Initializes the GraphBuilder with a retriever and LLM.
        
        Args:
            retriever: The retriever to use for fetching relevant documents.
            llm: The language model instance
        """
        self.nodes = RAGNodes()
        self.graph = None

    def build_graph(self):
        """Builds the RAG workflow graph.
        
        Returns:
            compiled graph instance.
        """
        #create state Graph
        builder = StateGraph(RAGState)

        #nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_response)

        #entry point
        builder.set_entry_point("retriever")

        #adding edges
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)

        #compile graph
        self.graph = builder.compile()
        return self.graph
    
    def run(self, question: str) -> str:
        """Runs the RAG Workflow
        
        Args:
            question: The input question for the RAG workflow.
            
        Returns:
            The generated answer from the RAG workflow.
        """
        if not self.graph:
            raise ValueError("Graph not built. Call build_graph first.")
        
        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)
   
