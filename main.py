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

# --- Configuration ---
# Using 127.0.0.1 is more stable than 'localhost' for local service connections
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen2.5vl:3b" 
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "your-key")
PINECONE_INDEX = "jarvis-index"

# --- Pinecone Initialization (Fixed Error 409) ---
pc = Pinecone(api_key=PINECONE_API_KEY)

# Get current indexes to avoid 'Resource already exists' error
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if PINECONE_INDEX not in existing_indexes:
    st.info(f"Creating index '{PINECONE_INDEX}'... This may take a minute.")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=3584, # Matching Qwen 2.5 VL embedding size
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    # Wait until the index is ready for operations
    while not pc.describe_index(PINECONE_INDEX).status['ready']:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX)

# --- Core RAG Functions ---

def get_embedding(text: str) -> List[float]:
    """Generate embedding using the modern Ollama /api/embed endpoint."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_MODEL, "input": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("embeddings", [[]])[0]
    except Exception as e:
        st.error(f"Ollama Connection Error: Ensure Ollama is running (ollama serve). Error: {e}")
        return []

def query_rag(query: str, top_k: int = 3) -> str:
    """Retrieve top matches from Pinecone based on query embedding."""
    query_embedding = get_embedding(query)
    if not query_embedding:
        return ""
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    context = "\n".join([match["metadata"].get("text", "") for match in results["matches"]])
    return context

def generate_response(query: str, context: str) -> str:
    """Send prompt with context to Ollama LLM."""
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer (be concise):"
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "No response generated")
    except Exception as e:
        return f"I'm sorry, I couldn't connect to my brain. Error: {e}"

# --- Main UI Interface ---
st.subheader("Chat")
with st.container():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

user_input = st.chat_input("Ask me something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Jarvis is thinking..."):
        context = query_rag(user_input)
        response = generate_response(user_input, context)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
        with st.expander("View source context"):
            st.write(context)

# --- Sidebar Management ---
st.sidebar.title("Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    # Split text into chunks for better retrieval
    chunks = [text[i:i+500] for i in range(0, len(text), 450)]
    st.sidebar.info(f"Processing {len(chunks)} chunks...")

    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if embedding:
            vectors_to_upsert.append((f"doc-{int(time.time())}-{i}", embedding, {"text": chunk}))
        
        # Batch upsert to improve speed
        if len(vectors_to_upsert) >= 50:
            index.upsert(vectors=vectors_to_upsert)
            vectors_to_upsert = []
            
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)
        
    st.sidebar.success("Digital Jarvis has updated its memory!")