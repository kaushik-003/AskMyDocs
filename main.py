"""AskMyDocs — CLI entry point for the RAG pipeline."""

from src.config.config import Config
from src.document_ingestion.documentprocessor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


def main():
    # 1. Load & split documents
    processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    print("Loading documents from default URLs…")
    raw_docs = processor.load_documents(Config.DEFAULT_URLS)
    chunks = processor.split_documents(raw_docs)
    print(f"  → {len(chunks)} chunks created.\n")

    # 2. Build vector store + retriever
    vs = VectorStore()
    vs.create_retriever(chunks)
    print("Vector store ready.\n")

    # 3. Build LangGraph workflow
    llm = Config.get_llm()
    builder = GraphBuilder(retriever=vs.retriever, llm=llm)
    builder.build_graph()

    # 4. Interactive Q&A loop
    print("AskMyDocs is ready! Type your question (or 'quit' to exit).\n")
    while True:
        question = input("You: ").strip()
        if not question or question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = builder.run(question)

        answer = (
            result.get("answer", "")
            if isinstance(result, dict)
            else getattr(result, "answer", str(result))
        )
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
