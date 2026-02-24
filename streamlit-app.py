"""AskMyDocs — Dark, vibrant RAG chat interface. Auto-loads from data/ folder."""

import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from src.config.config import Config
from src.document_ingestion.documentprocessor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AskMyDocs",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS — dark & vibrant ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Dark background ─────────────────────────────────── */
    .stApp {
        background: linear-gradient(160deg, #0a0a0f 0%, #111827 50%, #0f172a 100%);
        color: #e2e8f0;
    }

    /* ── Header ──────────────────────────────────────────── */
    .app-header {
        text-align: center;
        padding: 4rem 0 1.5rem;
    }
    .app-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa 0%, #6366f1 40%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.03em;
    }
    .app-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    /* ── Status badge ────────────────────────────────────── */
    .doc-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
    }

    /* ── Chat messages ───────────────────────────────────── */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.75rem !important;
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
        box-shadow: none !important;
    }

    /* user bubble */
    [data-testid="stChatMessageUser"] {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
    }

    /* assistant bubble */
    [data-testid="stChatMessageAssistant"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        border: 1px solid #2d3a5c !important;
        color: #e2e8f0 !important;
    }

    /* ── Chat input bar — shifted slightly left (60:40) ─── */
    .stChatInput {
        width: 100% !important;
        max-width: 100% !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }
    .stChatInput > div {
        border-radius: 24px !important;
        border: 1.5px solid #334155 !important;
        background: #1e293b !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.08) !important;
        padding: 0.15rem 0.5rem !important;
    }
    .stChatInput > div:focus-within {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.2) !important;
    }
    .stChatInput textarea {
        font-size: 0.95rem !important;
        color: #e2e8f0 !important;
    }
    .stChatInput textarea::placeholder {
        color: #64748b !important;
    }

    /* ── Buttons ──────────────────────────────────────────── */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s ease;
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.3) !important;
    }

    /* ── Spinner ──────────────────────────────────────────── */
    .stSpinner > div > div {
        border-top-color: #818cf8 !important;
    }

    /* ── Source expander ──────────────────────────────────── */
    details {
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        background: #1e293b !important;
        padding: 0.15rem 0.5rem !important;
    }
    details summary {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
    }
    details p, details span {
        color: #cbd5e1 !important;
    }

    /* ── Success / Warning / Error ────────────────────────── */
    .stAlert {
        border-radius: 12px !important;
    }

    /* ── Hide Streamlit chrome ────────────────────────────── */
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0


# ── Auto-ingest from data/ folder on first run ──────────────────────────────
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    """Load PDFs from data/ dir + URLs from data/urls.txt, embed and build graph."""
    processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    raw_docs = []
    data_dir = Path("data")

    # Load PDFs from data/
    pdf_files = list(data_dir.glob("*.pdf"))
    for pdf in pdf_files:
        raw_docs.extend(processor.load_from_pdf(pdf))

    # Load URLs from data/urls.txt
    urls_file = data_dir / "urls.txt"
    if urls_file.exists():
        urls = [
            line.strip()
            for line in urls_file.read_text().splitlines()
            if line.strip() and line.strip().startswith("http")
        ]
        for url in urls:
            raw_docs.extend(processor.load_from_url(url))

    if not raw_docs:
        return None, 0

    chunks = processor.split_documents(raw_docs)

    vs = VectorStore()
    vs.create_retriever(chunks)

    llm = Config.get_llm()
    builder = GraphBuilder(retriever=vs.retriever, llm=llm)
    builder.build_graph()

    return builder, len(chunks)


# ── Load data ────────────────────────────────────────────────────────────────
if not st.session_state.docs_loaded:
    with st.spinner("Loading knowledge base from `data/` folder…"):
        graph, count = load_knowledge_base()
        if graph:
            st.session_state.graph = graph
            st.session_state.doc_count = count
            st.session_state.docs_loaded = True


# ── Main area ────────────────────────────────────────────────────────────────

# Header (shown when chat is empty)
if not st.session_state.messages:
    st.markdown(
        """
        <div class="app-header">
            <h1>AskMyDocs</h1>
            <p>Your documents are loaded — just ask anything below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Status badge
if st.session_state.docs_loaded:
    st.markdown(
        f'<p style="text-align:center; margin-bottom:1.2rem;">'
        f'<span class="doc-badge">{st.session_state.doc_count} chunks ready</span></p>',
        unsafe_allow_html=True,
    )
elif not st.session_state.docs_loaded:
    st.error("No documents found in `data/` folder. Add PDFs or a `urls.txt` and restart.")

# Render chat history
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🔮"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.caption(f"**[{i}]** {src}")

# Chat input with Clear button
col_input, col_clear = st.columns([8, 1])
with col_input:
    prompt = st.chat_input("Ask a question about your documents…")
with col_clear:
    if st.button("🗑"):
        st.session_state.messages = []
        st.rerun()

if prompt:
    if not st.session_state.docs_loaded:
        st.warning("No documents loaded yet.")
    else:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant", avatar="🔮"):
            with st.spinner("Thinking…"):
                try:
                    result = st.session_state.graph.run(prompt)

                    answer = (
                        result.get("answer", "")
                        if isinstance(result, dict)
                        else getattr(result, "answer", str(result))
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
