"""Document processing module for loading and splitting documents."""

from typing import List, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_core.documents import Document
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)

class DocumentProcessor:
    """Handles Document loading and Processing."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Intialize the Document processor.
        
        Args:
            chunk_size: The size of each chunk.
            chunk_overlap: The overlap between chunks."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
            )
    def load_from_url(self, url: str) -> List[Document]:
        """Load documents from a URL."""
        loader = WebBaseLoader(url)
        return loader.load()
    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from a PDF directory."""
        loader = PyPDFDirectoryLoader(directory)
        return loader.load()
    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load documents from a text file."""
        loader = TextLoader(str(file_path), encoding='utf-8')
        return loader.load()
    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load documents from a PDF file."""
        loader = PyPDFLoader(str(file_path))
        return loader.load()
    
    def load_documents(self, sources: List[str]) -> List[Document]:
        """Load Documents from URLs, PDF directories, text files, or PDF files.
        
        Args:
            sources: List of URL, PDF directory, text file, or PDF file paths.
        Returns:
            List of loaded Documents.
        """
        docs: List[Document] = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
            else:
                path = Path(src)
                if path.is_dir():
                    docs.extend(self.load_from_pdf_dir(path))
                elif path.suffix.lower() == ".txt":
                    docs.extend(self.load_from_txt(path))
                elif path.suffix.lower() == ".pdf":
                    docs.extend(self.load_from_pdf(path))
                else:
                    raise ValueError(
                        f"Unsupported source type: {src}. "
                        "Use a URL, PDF directory, PDF file, or text file."
                    )
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks.
        
        Args:
            documents: List of Documents to split.
        Returns:
            List of split Documents."""
        return self.splitter.split_documents(documents)
    
    def process_url(self, urls: List[str]) -> List[Document]:
        """Process documents from URLs.
        
        Args:
            urls: List of URLs to process.
        Returns:
            List of processed Documents.
        """
        docs = self.load_documents(urls)
        return self.split_documents(docs)


                                 