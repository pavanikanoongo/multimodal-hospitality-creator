import streamlit as st
import io
from PIL import Image
from huggingface_hub import InferenceClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

# --- 1. THE FIX: CUSTOM EMBEDDING BRIDGE ---
class GeminiEmbeddingBridge(Embeddings):
    """
    Ensures ChromaDB receives one vector per document, 
    fixing the 'Unequal lengths' ValueError.
    """
    def __init__(self, model_name, api_key):
        self.client = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key
        )

    def embed_documents(self, texts):
        # We manually loop to ensure each text gets its own unique vector
        return [self.client.embed_query(text) for text in texts]

    def embed_query(self, text):
        return self.client.embed_query(text)

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Multimodal Hospitality Creator", layout="wide", page_icon="🏨")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #E63946; color: white; font-weight: bold; }
    .stTextInput>div>div>input { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏨 Multimodal Hospitality Creator")
st.caption("Building a synergy between text-to-narrative and text-to-visual models.")

# --- 3. SIDEBAR KEYS ---
with st.sidebar:
    st.header("🔑 API Credentials")
    gemini_key = st.text_input("Google Gemini API Key", type="password")
    hf_token = st.text_input("Hugging Face Token", type="password")
# --- 4. APPLICATION LOGIC ---
def launch_creator():
    # Knowledge base for RAG
    hospitality_data = [
        "Biophilic Luxury: Merges indoor greenery with high-tech automation and wood textures.",
        "Desert Futurism: Mirror-clad structures, subterranean suites, and brutalist sand-concrete.",
        "Heritage Revival: Restored stone courtyards, traditional jali work, and modern minimalist interiors.",
        "Alpine Minimalism: Floor-to-ceiling glass, floating fireplaces, and locally sourced pine."
    ]

    # Initialize the fixed embedding model
    # We use 'gemini-embedding-2-preview' for state-of-the-art 2026 performance
    embed_model = GeminiEmbeddingBridge(
        model_name="models/gemini-embedding-2-preview", 
        api_key=gemini_key
    )

    # User Input
    user_vision = st.text_input("Describe your hotel/resort concept:", 
                                placeholder="e.g., A floating glass villa in the Arctic...")

    if st.button("Generate Multimodal Concept"):
        if not user_vision:
            st.error("Please enter a concept description!")
            return

        with st.spinner("🔄 Retrieving standards, writing brief, and painting view..."):
            try:
                # A. Retrieval (RAG)
                # We build the vector store in-memory for the demo
                vdb = Chroma.from_texts(hospitality_data, embed_model)
                best_match = vdb.similarity_search(user_vision, k=1)[0].page_content

                # B. Text Synthesis (Gemini 1.5 Flash)
                llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=gemini_key)
                prompt = f"Using this style guide: '{best_match}', write a 3-sentence professional brief for: {user_vision}."
                response = llm.invoke(prompt)
                if isinstance(response.content, list):
                    # This extracts just the text from the JSON-style list
                    narrative = response.content[0].get("text", str(response.content))
                else:
                    narrative = response.content

                # C. Visual Synthesis (Hugging Face - FLUX)
                hf_client = InferenceClient(api_key=hf_token)
                img_prompt = f"Hyper-realistic architectural photography, {user_vision}, luxury resort, {best_match}, 8k, daylight, professional hospitality marketing."
                
                image_data = hf_client.text_to_image(img_prompt, model="black-forest-labs/FLUX.1-schnell")

                # --- RESULTS DISPLAY ---
                st.divider()
                left, right = st.columns([1, 1.2], gap="medium")
                
                with left:
                    st.subheader("📜 Architectural Brief")
                    st.write(narrative)
                    st.info(f"**Knowledge Base Reference:** {best_match}")
                
                with right:
                    st.subheader("🖼️ Concept Visualization")
                    st.image(image_data, use_container_width=True, caption="Visual generated via FLUX.1")
                    
            except Exception as error:
                st.error(f"Demo Error: {error}")

# Run app if keys exist
if gemini_key and hf_token:
    launch_creator()
else:
    st.warning("Please provide both API keys in the sidebar to start the demo.")