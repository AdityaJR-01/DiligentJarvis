# Digital Jarvis: RAG-Powered Assistant

This repository contains a Retrieval-Augmented Generation (RAG) chatbot designed for enterprise data assistance. The system integrates a local Large Language Model (LLM) with a cloud-based vector database to provide accurate, context-aware responses.

## Technical Specifications
- **Large Language Model:** Qwen 2.5 VL (via Ollama)
- **Vector Database:** Pinecone (Serverless)
- **Interface:** Streamlit
- **Protocol:** REST API (Ollama /api/generate and /api/embed)

## Key Architectural Decisions
- **Model Selection:** Qwen 2.5 VL was selected for its efficiency in local environments and high-dimensional embedding accuracy.
- **Optimization:** Implemented batch upserting for document ingestion to reduce network latency and ensure scalability.
- **Data Grounding:** The system uses a retrieval-first strategy, injecting verified context into the model prompt to minimize hallucinations.

## Setup Instructions
1. Ensure Ollama is installed and the Qwen 2.5 VL model is pulled: `ollama pull qwen2.5vl:3b`.
2. Configure your Pinecone API key in `.streamlit/secrets.toml`.
3. Install the required dependencies: `pip install -r requirements.txt`.
4. Run the application: `streamlit run main.py`.