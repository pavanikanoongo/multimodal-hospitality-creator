# Multimodal Hospitality Creator

An AI-driven design orchestration platform that generates professional architectural briefs and high-fidelity visualizations using RAG.

## Features
- **RAG Integration:** Uses ChromaDB and Gemini Embeddings to ground AI generation in professional hospitality standards.
- **Multimodal Synthesis:** Combines text-to-narrative (Gemini 3 Flash) and text-to-image (FLUX.1) workflows.
- **Enterprise Ready:** Features custom embedding wrappers and secure API handling.

## Tech Stack
- **Language:** Python
- **Framework:** Streamlit
- **AI Models:** Google Gemini 3, FLUX.1 (Black Forest Labs)
- **Vector DB:** ChromaDB
- **Orchestration:** LangChain

## Installation
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`