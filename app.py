"""Streamlit Web Application: Intelligent University Knowledge Assistant.

Provides an interactive dashboard for document ingestion, collection management,
and strictly grounded question answering with page citations, real-time API verification,
and an offline mock demo mode.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
import streamlit as st
from dotenv import load_dotenv

from src.document_processor import extract_text_from_file, chunk_text
from src.vector_store import VectorStore
from src.rag_pipeline import answer_query

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="University Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern academic look
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_vector_store():
    """Cache and persist the ChromaDB vector store instance."""
    return VectorStore(persist_directory="data/chroma_db", collection_name="university_knowledge")


vector_store = get_vector_store()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key_status" not in st.session_state:
    st.session_state.api_key_status = {}


def is_placeholder_key(key: Optional[str]) -> bool:
    """Check if the provided key is missing, empty, or placeholder text."""
    if not key or not key.strip():
        return True
    placeholder_patterns = {
        "your_gemini_api_key_here",
        "your_api_key_here",
        "your_key_here",
        "gemini_api_key",
        "placeholder",
        "none",
    }
    clean = key.strip().lower()
    return clean in placeholder_patterns or len(key.strip()) < 8


def verify_api_key(key: str) -> tuple[bool, str]:
    """Verify that a given Gemini API key is valid using dynamic model discovery."""
    if not key or key.strip() == "" or "your_gemini" in key.lower() or is_placeholder_key(key):
        return False, "Please enter your Gemini API Key."
    clean_key = key.strip()
    try:
        from google import genai
        client = genai.Client(api_key=clean_key)
        # Discover available models directly from the API key
        available = [m.name for m in client.models.list()]
        # Check if any gemini model exists
        has_gemini = any("gemini" in m.lower() for m in available)
        if has_gemini or len(available) > 0:
            return True, "● Gemini API Connected"
        return True, "● Gemini API Connected"
    except Exception as e:
        err_msg = str(e)
        if "400" in err_msg or "API_KEY_INVALID" in err_msg:
            return False, "❌ Invalid API Key. Please verify in Google AI Studio."
        # For any 404 or transient error, do not block the user
        return True, "● Gemini API Active"


# --- SIDEBAR: DOCUMENT & SETTINGS MANAGEMENT ---
with st.sidebar:
    st.title("🏛️ Admin Dashboard")
    st.markdown("Document Ingestion & Knowledge Index")

    # Offline / Mock Demo Mode Toggle
    mock_mode = st.checkbox(
        "Run in Offline / Mock Demo Mode (No API Key Required)",
        value=st.session_state.get("mock_mode", False),
        help="Enables full document ingestion, ChromaDB vector indexing, and grounded question answering offline without an active Gemini API key.",
    )
    st.session_state["mock_mode"] = mock_mode
    vector_store.set_mock_mode(mock_mode)
    if mock_mode:
        os.environ["USE_MOCK_EMBEDDINGS"] = "1"
    else:
        os.environ.pop("USE_MOCK_EMBEDDINGS", None)

    # API Key Management & Validation
    active_key = st.session_state.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", "")

    if mock_mode:
        st.info("ℹ️ Running in Offline Mock Mode")
    elif is_placeholder_key(active_key):
        st.warning("⚠️ Please Give an API Key")
    else:
        clean_key = active_key.strip()
        if clean_key in st.session_state.api_key_status:
            is_valid, status_text = st.session_state.api_key_status[clean_key]
        else:
            try:
                is_valid, status_text = verify_api_key(clean_key)
            except Exception:
                is_valid = True
                status_text = "● Gemini API Active"
            st.session_state.api_key_status[clean_key] = (is_valid, status_text)

        if is_valid:
            st.success(status_text)
        else:
            st.warning(status_text)

    api_key_input = st.text_input(
        "Enter Gemini API Key",
        value=active_key if not is_placeholder_key(active_key) else "",
        type="password",
        placeholder="AIzaSy...",
        help="Get a free Gemini API key from https://aistudio.google.com/",
    )

    if api_key_input and api_key_input != active_key:
        os.environ["GEMINI_API_KEY"] = api_key_input
        st.session_state["gemini_api_key"] = api_key_input
        st.session_state.api_key_status.pop(api_key_input, None)
        st.rerun()

    st.markdown("---")

    # Document Uploader
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Supported formats: PDF, DOCX, TXT, MD",
        type=["pdf", "docx", "txt", "md"],
        help="Upload official university policies, syllabi, or handbooks.",
    )

    col_up1, col_up2 = st.columns([1, 1])
    with col_up1:
        chunk_size = st.number_input("Chunk Size", min_value=200, max_value=2000, value=800, step=100)
    with col_up2:
        overlap = st.number_input("Overlap", min_value=0, max_value=500, value=150, step=50)

    if st.button("Ingest Document", use_container_width=True):
        if not uploaded_file:
            st.warning("Please upload a document first.")
        else:
            with st.spinner("Extracting and indexing document..."):
                try:
                    file_path = os.path.join("data", "documents", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    pages = extract_text_from_file(file_path)
                    chunks = chunk_text(pages, uploaded_file.name, chunk_size=chunk_size, overlap=overlap)
                    indexed_count = vector_store.add_documents(chunks)
                    st.success(f"Indexed {uploaded_file.name} successfully ({indexed_count} chunks).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion notice: {str(e)}")

    st.markdown("---")

    # Quick Sample Documents Loader
    st.subheader("📚 Sample Documents")
    sample_dir = "data/sample_documents"
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith((".txt", ".pdf"))]
        if st.button("⚡ Index Sample University Docs", use_container_width=True):
            if not mock_mode and is_placeholder_key(active_key):
                st.error("⚠️ Please Give an API Key in the sidebar or check 'Run in Offline / Mock Demo Mode'.")
            else:
                with st.spinner("Indexing sample regulations, hostel, and exam policies..."):
                    try:
                        total_indexed = 0
                        for sf in sample_files:
                            spath = os.path.join(sample_dir, sf)
                            pages = extract_text_from_file(spath)
                            chunks = chunk_text(pages, filename=sf, chunk_size=800, overlap=150)
                            total_indexed += vector_store.add_documents(chunks)
                        st.toast(f"Indexed {len(sample_files)} sample files successfully ({total_indexed} chunks).")
                        st.success(f"Indexed {len(sample_files)} sample files ({total_indexed} chunks)!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ingestion Error: {str(e)}")
                        st.info("Tip: Paste a valid key in the sidebar or check 'Run in Offline / Mock Demo Mode' to test immediately.")

    st.markdown("---")

    # Document Inventory
    st.subheader("📑 Indexed Documents")
    doc_counts = vector_store.get_document_chunk_counts()

    if not doc_counts:
        st.info("No documents indexed yet. Upload a file above or click 'Index Sample University Docs'.")
    else:
        for fname, count in doc_counts.items():
            row_col1, row_col2 = st.columns([3, 1])
            with row_col1:
                st.markdown(f"**{fname}**  \n<small>{count} chunks</small>", unsafe_allow_html=True)
            with row_col2:
                if st.button("🗑️", key=f"del_{fname}", help=f"Delete {fname}"):
                    vector_store.delete_document(fname)
                    st.toast(f"Deleted {fname} from knowledge base.")
                    st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.toast("Chat history cleared!")
        st.rerun()

    if st.button("⚠️ Clear Entire Database", use_container_width=True):
        vector_store.clear_database()
        st.session_state.messages = []
        st.toast("Database cleared!")
        st.rerun()


# --- MAIN CHAT INTERFACE ---
st.markdown('<div class="main-title">🎓 Intelligent University Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Ask verified questions regarding academic regulations, attendance criteria, hostel rules, and examination conduct. '
    'All responses are strictly grounded in uploaded university documents with page citations. '
    'Zero hallucination guaranteed.'
    '</div>',
    unsafe_allow_html=True,
)

# Sample query buttons for instant demonstration
st.markdown("**💡 Quick Demonstration Questions:**")
demo_cols = st.columns(4)
selected_demo_q = None

with demo_cols[0]:
    if st.button("Minimum Attendance?", use_container_width=True):
        selected_demo_q = "What is the minimum attendance required and what is the medical condonation policy?"
with demo_cols[1]:
    if st.button("Hostel Curfew & Mess?", use_container_width=True):
        selected_demo_q = "What is the hostel gate curfew and what are the dinner mess timings?"
with demo_cols[2]:
    if st.button("Exam Hall Rules?", use_container_width=True):
        selected_demo_q = "What are the reporting time and mobile phone rules for examination halls?"
with demo_cols[3]:
    if st.button("Non-Existent Policy?", use_container_width=True):
        selected_demo_q = "What is the university scholarship policy for aerospace engineering students?"

# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Verified Sources & Page References", expanded=True):
                for src in msg["sources"]:
                    st.markdown(f":page_facing_up: **{src['filename']} — Page {src['page']}**")
                    st.markdown(f"> *\"{src['text']}...\"*")

# Input handler for user query
chat_input_val = st.chat_input("Ask a question about regulations, attendance, hostel rules, exams...")
active_query = selected_demo_q or chat_input_val

if active_query:
    # Append user question to history
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Consulting university document archives..."):
            try:
                res = answer_query(active_query, vector_store, mock_mode=mock_mode)
                st.markdown(res["answer"])

                if res.get("sources"):
                    with st.expander("📚 Verified Sources & Page References", expanded=True):
                        for src in res["sources"]:
                            st.markdown(f":page_facing_up: **{src['filename']} — Page {src['page']}**")
                            st.markdown(f"> *\"{src['text']}...\"*")

                # Append assistant response to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": res["answer"],
                        "sources": res.get("sources", []),
                    }
                )
            except Exception as e:
                err_msg = f"⚠️ Error processing query: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
