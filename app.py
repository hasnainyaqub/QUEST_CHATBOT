import json
import streamlit as st
import os
import pickle
import faiss
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# ----------------------------
# Modern UI Styling (Light Mode)
# ----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f7f9fc;
}

h1 {
    color: #1f2a44;
    font-weight: 700;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff, #f0f4ff);
    border-right: 1px solid #e6e9f0;
}

section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #2c3e50;
}

input[type="text"] {
    background-color: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #dce1eb !important;
    padding: 12px !important;
}

button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 10px 18px !important;
    border: none !important;
    font-weight: 600 !important;
}

.user-bubble {
    background: #e0e7ff;
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 12px;
    font-weight: 500;
}

.ai-bubble {
    background: #ffffff;
    padding: 16px;
    border-radius: 16px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.05);
    border-left: 5px solid #6366f1;
    line-height: 1.6;
}

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Load FAISS + metadata
# ----------------------------
@st.cache_resource
def load_vectorstore():
    index = faiss.read_index("faiss.index")
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


index, metadata = load_vectorstore()


@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )


embedder = load_embedder()


# ----------------------------
# Retrieval
# ----------------------------
def retrieve(query, k=5):
    q_emb = embedder.embed_query(query)
    q_emb = np.array([q_emb], dtype="float32")

    _, indices = index.search(q_emb, k)

    return [
        metadata["docstore"][metadata["index_to_docstore_id"][int(i)]]
        for i in indices[0]
    ]


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="QUEST RAG App", layout="wide")

st.title("📄 QUEST RAG Assistant")
st.markdown(
    "Ask questions directly from **official QUEST documents** using an AI-powered retrieval system."
)

st.sidebar.header("⚙️ Model Settings")

provider = "Groq"

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-safeguard-20b",
        "openai/gpt-oss-20b",
        "groq/compound",
        "groq/compound-mini",
        "llama-3.3-70b-versatile"
    ]
)

api_key = st.secrets['GROQ_API_KEY']

headers = {
       "authorization": f"Bearer {api_key}",
        "content-type": "application/json"
}

st.sidebar.markdown("### API Sources")
st.sidebar.markdown(
    "Recommended to use **Groq API** with **gpt-oss-120b** for best results. \n"
    "- **Groq API**  \n"
    "https://console.groq.com/keys"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='footer'>Built with FAISS + LangChain + Groq</div>",
    unsafe_allow_html=True
)


# ----------------------------
# Load LLM
# ----------------------------
def load_llm(provider, model_name, api_key):
    os.environ["GROQ_API_KEY"] = api_key
    return ChatGroq(
        model=model_name,
        temperature=0
    )


# ----------------------------
# RAG Answer
# ----------------------------
def rag_answer(query, llm):
    docs = retrieve(query)

    if not docs:
        return "Not found in documents."

    context = "\n\n".join(
        [
            f"Source: {d.metadata.get('source', 'N/A')} | Page: {d.metadata.get('page', 'N/A')}\n{d.page_content}"
            for d in docs
        ]
    )

    prompt = f"""
You are an assistant answering questions using official QUEST documents.

Use ONLY the context provided below.
Answer the question clearly and concisely.
If the information is not explicitly present in the context, reply exactly:
"Not found in documents."

Context is related to:
QUEST – Quaid-e-Awam University of Engineering, Science and Technology, Nawabshah.

Context:
{context}

Question:
{query}
"""

    return llm.invoke(prompt).content


# ----------------------------
# Main Chat
# ----------------------------
st.markdown("### 💬 Ask a Question")

query = st.text_input(
    "Type your question here",
    placeholder="e.g. What is the admission criteria for BE programs?"
)

if st.button("Ask"):
    llm = load_llm(provider, model_name, api_key)

    with st.spinner("Thinking..."):
        answer = rag_answer(query, llm)

    st.markdown(
        f"<div class='user-bubble'>🧑 <b>You:</b><br>{query}</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div class='ai-bubble'>🤖 <b>AI Assistant:</b><br>{answer}</div>",
        unsafe_allow_html=True
    )
