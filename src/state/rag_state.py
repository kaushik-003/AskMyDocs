"""RAG state definition for LangGraph."""

from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document

class RAGState(BaseModel):
    """State for RAG operations."""

    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""
  
