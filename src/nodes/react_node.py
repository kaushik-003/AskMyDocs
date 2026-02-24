"""LangGraph Nodes for RAG Workflow + ReAct agent inside generate-context node."""

from typing import List, Optional
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGReActNodes:
    """Node functions for RAG workflow with ReAct agent."""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy initialization of agent

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node."""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs,
        )

    # ── Tools ────────────────────────────────────────────────────────────────
    def _build_tools(self) -> List[Tool]:
        """Builds tools for the agent."""

        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No relevant documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                metadata = d.metadata if hasattr(d, "metadata") else {}
                title = metadata.get("title") or metadata.get("source") or f"doc {i}"
                merged.append(f"[{i}] {title}: {d.page_content}")
            return "\n\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            description=(
                "Use this tool to retrieve relevant documents based on the user query. "
                "Input should be the user query. Output will be retrieved documents."
            ),
            func=retriever_tool_fn,
        )

        wiki_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en"),
        )
        wikipedia_tool = Tool(
            name="wikipedia",
            description=(
                "Use this tool to query Wikipedia for general knowledge questions. "
                "Input should be the user query."
            ),
            func=wiki_tool.run,
        )

        return [retriever_tool, wikipedia_tool]

    # ── Agent ────────────────────────────────────────────────────────────────
    def _build_agent(self):
        """Builds a ReAct agent with retriever + Wikipedia tools."""
        tools = self._build_tools()
        self._agent = create_react_agent(
            model=self.llm,
            tools=tools,
            prompt="You are a helpful RAG agent. "
                   "Prefer the 'retriever' tool for user-provided docs; "
                   "use the 'wikipedia' tool for general knowledge. "
                   "Return only the final useful answer.",
        )

    def generate_response(self, state: RAGState) -> RAGState:
        """Generates a response using the ReAct agent."""
        if not self._agent:
            self._build_agent()

        result = self._agent.invoke(
            {"messages": [HumanMessage(content=state.question)]}
        )
        messages = result.get("messages", [])
        response: Optional[str] = None
        if messages:
            response = getattr(messages[-1], "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response or "Sorry, I couldn't generate a response.",
        )
            

