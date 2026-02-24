"""AskMyDocs — A minimal, Claude-inspired RAG chat interface."""

import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.config.config import Config
from src.document_ingestion.documentprocessor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AskMyDocs",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS — minimal, warm, Claude-like ──────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #FAF9F6;
    }

    /* ── Header ────────────────────────────────────────────── */
    .app-header {
        text-align: center;
        padding: 2.5rem 0 1rem;
    }
    .app-header h1 {
        font-size: 1.85rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #6b6b6b;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }

    /* ── Chat messages ─────────────────────────────────────── */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.6rem !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        box-shadow: none !important;
    }

    /* user bubble */
    [data-testid="stChatMessageUser"] {
        background: #EDEDEC !important;
    }

    /* assistant bubble */
    [data-testid="stChatMessageAssistant"] {
        background: #ffffff !important;
        border: 1px solid #e8e8e8 !important;
    }

    /* ── Chat input ────────────────────────────────────────── */
    .stChatInput > div {
        border-radius: 24px !important;
        border: 1.5px solid #d4d4d4 !important;
        background: #ffffff !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
        padding: 0.15rem 0.5rem !important;
    }
    .stChatInput > div:focus-within {
        border-color: #D97757 !important;
        box-shadow: 0 0 0 2px rgba(217,119,87,0.15) !important;
    }
    .stChatInput textarea {
        font-size: 0.95rem !important;
    }

    /* ── Sidebar ───────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #F5F4F0;
        border-right: 1px solid #e8e8e8;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }

    /* ── Buttons ───────────────────────────────────────────── */
    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 0.45rem 1.1rem;
        transition: all 0.15s ease;
    }
    div.stButton > button[kind="primary"],
    div.stButton > button:first-child {
        background-color: #D97757 !important;
        color: white !important;
        border: none !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button:first-child:hover {
        background-color: #c4613f !important;
    }

    /* ── Status / spinner ──────────────────────────────────── */
    .stSpinner > div > div {
        border-top-color: #D97757 !important;
    }

    /* ── Pill / badge for doc count ────────────────────────── */
    .doc-badge {
        display: inline-block;
        background: #D97757;
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        margin-left: 0.4rem;
    }

    /* ── Source expander ────────────────────────────────────── */
    details {
        border: 1px solid #e8e8e8 !important;
        border-radius: 12px !important;
        background: #fafafa !important;
        padding: 0.15rem 0.5rem !important;
    }
    details summary {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #888 !important;
    }

    /* hide default Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# ── Sidebar: data ingestion ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Data Sources")

    source_type = st.radio(
        "Choose source type",
        ["URLs", "PDF Files", "PDF Directory", "Text Files"],
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded_files = None
    url_input = ""
    dir_path = ""

    if source_type == "URLs":
        url_input = st.text_area(
            "Enter URLs (one per line)",
            height=120,
            placeholder="https://example.com/article\nhttps://...",
        )
    elif source_type in ("PDF Files", "Text Files"):
        allowed = ["pdf"] if source_type == "PDF Files" else ["txt"]
        uploaded_files = st.file_uploader(
            f"Upload {source_type.lower()}",
            type=allowed,
            accept_multiple_files=True,
        )
    elif source_type == "PDF Directory":
        dir_path = st.text_input("Directory path", placeholder="./data")

    st.markdown("---")

    # ── Settings
    st.markdown("## ⚙️ Settings")
    chunk_size = st.slider("Chunk size", 200, 2000, Config.CHUNK_SIZE, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 200, Config.CHUNK_OVERLAP, 10)
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                              index=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"].index(Config.LLM_MODEL))

    st.markdown("---")

    # ── Ingest button
    ingest_btn = st.button("🚀  Ingest Documents", use_container_width=True)

    if ingest_btn:
        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        raw_docs = []

        try:
            with st.spinner("Loading documents…"):
                if source_type == "URLs" and url_input.strip():
                    urls = [u.strip() for u in url_input.strip().splitlines() if u.strip()]
                    for url in urls:
                        raw_docs.extend(processor.load_from_url(url))

                elif source_type == "PDF Files" and uploaded_files:
                    import tempfile
                    for f in uploaded_files:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        tmp.write(f.read())
                        tmp.close()
                        raw_docs.extend(processor.load_from_pdf(tmp.name))

                elif source_type == "Text Files" and uploaded_files:
                    import tempfile
                    for f in uploaded_files:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
                        tmp.write(f.read().decode("utf-8"))
                        tmp.close()
                        raw_docs.extend(processor.load_from_txt(tmp.name))

                elif source_type == "PDF Directory" and dir_path.strip():
                    raw_docs.extend(processor.load_from_pdf_dir(dir_path.strip()))

                else:
                    st.warning("Please provide at least one source.")

            if raw_docs:
                with st.spinner("Splitting & embedding…"):
                    chunks = processor.split_documents(raw_docs)

                    vs = VectorStore()
                    vs.create_retriever(chunks)

                    llm = Config.get_llm(model=model_name)
                    builder = GraphBuilder(retriever=vs.retriever, llm=llm)
                    builder.build_graph()

                    st.session_state.graph = builder
                    st.session_state.docs_loaded = True
                    st.session_state.doc_count = len(chunks)
                    st.session_state.vector_store = vs

                st.success(f"Ingested **{len(chunks)}** chunks!")

        except Exception as e:
            st.error(f"Ingestion failed: {e}")

    # ── Clear chat
    if st.button("🗑  Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main area ────────────────────────────────────────────────────────────────

# Header
if not st.session_state.messages:
    st.markdown(
        """
        <div class="app-header">
            <h1>AskMyDocs</h1>
            <p>Upload documents in the sidebar, then ask anything.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Status pill
if st.session_state.docs_loaded:
    st.markdown(
        f'<p style="text-align:center; margin-bottom:1rem;">'
        f'<span class="doc-badge">{st.session_state.doc_count} chunks ready</span></p>',
        unsafe_allow_html=True,
    )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "📄"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.caption(f"**[{i}]** {src}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    if not st.session_state.docs_loaded:
        st.warning("Please ingest documents first using the sidebar.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="📄"):
            with st.spinner("Thinking…"):
                try:
                    result = st.session_state.graph.run(prompt)

                    answer = result.get("answer", "") if isinstance(result, dict) else (
                        result.answer if hasattr(result, "answer") else str(result)
                    )
                    sources = []
                    retrieved = (
                        result.get("retrieved_docs", [])
                        if isinstance(result, dict)
                        else getattr(result, "retrieved_docs", [])
                    )
                    for doc in retrieved[:5]:
                        meta = doc.metadata if hasattr(doc, "metadata") else {}
                        label = meta.get("source") or meta.get("title") or "document"
                        page = meta.get("page")
                        if page is not None:
                            label += f" (p. {page})"
                        sources.append(label)

                except Exception as e:
                    answer = f"Something went wrong: {e}"
                    sources = []

            # Typewriter effect
            placeholder = st.empty()
            displayed = ""
            for char in answer:
                displayed += char
                placeholder.markdown(displayed + "▌")
                time.sleep(0.008)
            placeholder.markdown(answer)

            if sources:
                with st.expander("View sources"):
                    for i, src in enumerate(sources, 1):
                        st.caption(f"**[{i}]** {src}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
