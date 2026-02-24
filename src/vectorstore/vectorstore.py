"""vector store module for document embedding and retrieval."""


from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class VectorStore:
    """Handles vector store creation and retrieval."""
    
    def __init__(self):
        self.embedding = OpenAIEmbeddings()
        self.vector_store = None
        self.retriever = None

    def create_retriever(self, documents: List[Document]):
        """creates a vector store from the documents
        
        Args:
            documents: List of Documents to be embedd.
        """
        
        self.vector_store = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vector_store.as_retriever()
    def retrieve(self, query: str, top_k: int = 4) -> List[Document]:
        """Retrieves relevant documents based on a query.
        
        Args:
            query: search query.
            k: number of top relevant documents to retrieve.
        Returns:
            List of relevant Documents.    
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Call create_retriever first.")
        
        return self.retriever.invoke(query)