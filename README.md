# Digital Jarvis: RAG-Powered Assistant
 
This repository was mainly conceived to complete as a part of a one hour challenge, but was left incomplete. It contains a Retrieval-Augmented Generation (RAG) chatbot designed for enterprise data assistance. The system integrates a local Large Language Model (LLM) with a cloud-based vector database to provide accurate, context-aware responses.
 
## Technical Specifications
- **Large Language Model:** Qwen 2.5 VL (via Ollama) — chat generation
- **Embedding Model:** nomic-embed-text (via Ollama) — 768-dim, purpose-built for retrieval
- **Vector Database:** Pinecone (Serverless)
- **Interface:** Streamlit
- **Protocol:** REST API (Ollama /api/generate and /api/embed)
## Key Architectural Decisions
- **Model Selection:** Qwen 2.5 VL handles chat generation; a dedicated embedding model (nomic-embed-text) is used for retrieval instead, since general-purpose chat models aren't built for embedding quality or speed.
- **Optimization:** Implemented batch upserting for document ingestion to reduce network latency and ensure scalability.
- **Data Grounding:** The system uses a retrieval-first strategy, injecting verified context into the model prompt to minimize hallucinations.
## Setup Instructions
1. Ensure Ollama is installed and running (`ollama serve`), then pull both models:
```
   ollama pull qwen2.5vl:3b
   ollama pull nomic-embed-text
```
2. Configure Pinecone: create `.streamlit/secrets.toml` with:
```
   PINECONE_API_KEY = "your-actual-key"
   PINECONE_INDEX = "your-index-name"   # optional — defaults to "diligent-index"
```
3. Install the required dependencies: `pip install -r requirements.txt`.
4. Run the application: `streamlit run main.py`.

