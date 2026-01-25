import streamlit as st
import requests
import json
from pinecone import Pinecone, ServerlessSpec
from typing import List
import time

# --- Page Configuration ---
st.set_page_config(page_title="Digital Jarvis", layout="wide")
st.title("Digital Jarvis - RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "kb_documents" not in st.session_state:
    st.session_state.kb_documents = []  # [{"name": ..., "chunks": ...}, ...]
if "processed_signatures" not in st.session_state:
    st.session_state.processed_signatures = set()  # "{name}-{size}" of files already ingested

# --- Configuration ---
# Using 127.0.0.1 is more stable than 'localhost' for local service connections
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_CHAT_MODEL = "qwen2.5vl:3b"
# Dedicated embedding model instead of routing embeddings through the chat model:
# faster per chunk, and its output dimension is actually documented (768),
# unlike qwen2.5vl's pooled hidden state.
OLLAMA_EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "your-key")
PINECONE_INDEX = st.secrets.get("PINECONE_INDEX", "diligent-index")

# --- Pinecone Initialization (Fixed Error 409) ---
# Cached so this setup/check only runs once per session instead of on every
# Streamlit rerun (which happens on every chat message and every upload).
@st.cache_resource
def init_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if PINECONE_INDEX not in existing_indexes:
        st.info(f"Creating index '{PINECONE_INDEX}'... This may take a minute.")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBED_DIM,  # Matching nomic-embed-text's output size
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        while not pc.describe_index(PINECONE_INDEX).status['ready']:
            time.sleep(1)

    return pc.Index(PINECONE_INDEX)

index = init_pinecone_index()

def sync_kb_from_pinecone():
    """Reconstruct the knowledge base doc list from what's actually stored in Pinecone,
    so a fresh session reflects documents uploaded in a previous one instead of
    showing empty even though the data is still there."""
    doc_counts = {}
    try:
        for id_page in index.list():
            if not id_page:
                continue
            fetched = index.fetch(ids=list(id_page))
            for vec_id, vec_data in fetched["vectors"].items():
                meta = vec_data.get("metadata") or {}
                source = meta.get("source", "unknown source")
                doc_counts[source] = doc_counts.get(source, 0) + 1
    except Exception as e:
        st.sidebar.warning(f"Couldn't sync existing knowledge base from Pinecone: {e}")
        return

    st.session_state.kb_documents = [
        {"name": name, "chunks": count} for name, count in sorted(doc_counts.items())
    ]

if not st.session_state.get("kb_synced", False):
    sync_kb_from_pinecone()
    st.session_state.kb_synced = True

# --- Core RAG Functions ---

def get_embedding(text: str) -> List[float]:
    """Generate embedding using the modern Ollama /api/embed endpoint."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("embeddings", [[]])[0]
    except Exception as e:
        st.error(f"Ollama Connection Error: Ensure Ollama is running (ollama serve). Error: {e}")
        return []

def query_rag(query: str, top_k: int = 3):
    """Retrieve top matches from Pinecone based on query embedding.
    Returns (context_string_for_prompting, raw_matches_for_display)."""
    query_embedding = get_embedding(query)
    if not query_embedding:
        return "", []
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    matches = results.get("matches", [])
    context = "\n".join(match["metadata"].get("text", "") for match in matches)
    return context, matches

def generate_response(query: str, context: str) -> str:
    """Send prompt with context to Ollama LLM."""
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer (be concise):"
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_CHAT_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "No response generated")
    except Exception as e:
        return f"I'm sorry, I couldn't connect to my brain. Error: {e}"

# --- Main UI Interface ---

def render_retrieved_chunks(matches):
    """Formatted display of what was actually retrieved, instead of a raw text dump."""
    if not matches:
        st.caption("No relevant context found in the knowledge base.")
        return
    with st.expander(f"View {len(matches)} retrieved chunk(s)"):
        for i, match in enumerate(matches, 1):
            meta = match.get("metadata", {})
            score = match.get("score", 0)
            st.markdown(f"**{i}. {meta.get('source', 'unknown source')}** — relevance {score:.2f}")
            st.text(meta.get("text", ""))
            if i < len(matches):
                st.divider()

st.subheader("Chat")
with st.container():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                render_retrieved_chunks(msg.get("matches", []))

user_input = st.chat_input("Ask me something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Jarvis is thinking..."):
        context, matches = query_rag(user_input)
        response = generate_response(user_input, context)

    st.session_state.messages.append({"role": "assistant", "content": response, "matches": matches})
    with st.chat_message("assistant"):
        st.write(response)
        render_retrieved_chunks(matches)

# --- Sidebar: Chat controls ---
st.sidebar.title("Chat")
if st.sidebar.button("🆕 New Chat", disabled=not st.session_state.messages):
    st.session_state.messages = []
    st.rerun()

st.sidebar.divider()

# --- Sidebar: Knowledge Base ---
st.sidebar.title("Knowledge Base")

if st.session_state.kb_documents:
    total_chunks = sum(doc["chunks"] for doc in st.session_state.kb_documents)
    st.sidebar.caption(f"{len(st.session_state.kb_documents)} document(s), {total_chunks} chunks")
    for doc in st.session_state.kb_documents:
        st.sidebar.markdown(f"📄 {doc['name']} — {doc['chunks']} chunks")

    if st.sidebar.button("🗑️ Clear knowledge base"):
        index.delete(delete_all=True)
        st.session_state.kb_documents = []
        st.session_state.processed_signatures = set()
        st.sidebar.success("Knowledge base cleared.")
        st.rerun()
else:
    st.sidebar.caption("No documents yet — upload one below to get started.")

uploaded_files = st.sidebar.file_uploader(
    "Upload document(s)", type=["txt"], accept_multiple_files=True
)

if uploaded_files:
    new_files = [
        f for f in uploaded_files
        if f"{f.name}-{f.size}" not in st.session_state.processed_signatures
    ]

    for uploaded_file in new_files:
        text = uploaded_file.read().decode("utf-8")
        # Split text into chunks for better retrieval
        chunks = [text[i:i+500] for i in range(0, len(text), 450)]
        st.sidebar.info(f"Processing '{uploaded_file.name}' ({len(chunks)} chunks)...")

        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if embedding:
                vectors_to_upsert.append((
                    f"doc-{uploaded_file.name}-{int(time.time())}-{i}",
                    embedding,
                    {"text": chunk, "source": uploaded_file.name}
                ))

            # Batch upsert to improve speed
            if len(vectors_to_upsert) >= 50:
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []

        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)

        st.session_state.kb_documents.append({"name": uploaded_file.name, "chunks": len(chunks)})
        st.session_state.processed_signatures.add(f"{uploaded_file.name}-{uploaded_file.size}")

    if new_files:
        st.sidebar.success(f"Added {len(new_files)} document(s) to Jarvis's memory!")
        st.rerun()