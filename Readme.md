# 📄 QUEST RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to ask questions directly from official QUEST (Quaid-e-Awam University of Engineering, Science and Technology) documents.

The app uses FAISS for vector search, HuggingFace embeddings, and Groq LLMs, wrapped in a modern Streamlit UI with a clean light-mode AI chat interface.

---

## 🚀 Features

- 🔍 Semantic search over official QUEST documents
- 📚 FAISS vector database for fast retrieval
- 🤖 Groq-powered LLM responses
- 🧠 BAAI bge-base-en-v1.5 embeddings
- 💬 Modern AI chat-style UI (light mode, vibrant design)
- 🧾 Source-aware context injection
- ⚡ Fast, accurate, and production-ready RAG pipeline

---

## 🧠 How It Works (RAG Pipeline)

1. **Documents Ingestion**  
   QUEST prospectus and official PDFs are chunked and embedded

2. **Vector Storage**  
   Embeddings are stored in a FAISS index

3. **Query Processing**  
   User question is embedded using the same model

4. **Retrieval**  
   FAISS returns the most relevant document chunks

5. **Generation**  
   Groq LLM generates answers strictly from retrieved context

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Vector Store** | FAISS |
| **Embeddings** | HuggingFace BAAI/bge-base-en-v1.5 |
| **LLM Provider** | Groq |
| **Frameworks** | LangChain |
| **Language** | Python |

---

## 📂 Project Structure

```
.
├── app.py                # Main Streamlit application
├── faiss.index           # FAISS vector index
├── metadata.pkl          # Document metadata & mappings
├── README.md             # Project documentation
├── RAG.ipynb             # Jupyter Notebook for RAG pipeline
└── requirements.txt      # Python dependencies
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/quest-rag-assistant.git
cd quest-rag-assistant
```

### 2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Groq API Key

Create the file `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open in your browser:
```
http://localhost:8501
```

---

## 🧪 Example Queries

- What is the admission criteria for BE programs?
- What is the fee structure for undergraduate students?
- Which departments are offered at QUEST?
- What are the eligibility requirements?

---

## 🔒 Safety & Accuracy

- ✅ The assistant **ONLY** uses retrieved document context
- ✅ If information is missing, it responds with:  
  ```
  Not found in documents.
  ```
- ✅ No hallucinated answers are generated

---

## 🎨 UI Highlights

- Light-mode focused design
- Vibrant modern AI interface
- Chat-style question and answer flow
- Clean typography and spacing

---

## 📈 Future Improvements

- [ ] Streaming token-by-token responses
- [ ] Conversation memory
- [ ] Clickable citations per source
- [ ] Hybrid retrieval (BM25 + FAISS)
- [ ] Docker deployment
- [ ] Multi-PDF support

---

## 👤 Author

**Hasnain Yaqub**  
AI / ML & Generative AI Practitioner

---

## ⭐ Acknowledgements

- [FAISS](https://github.com/facebookresearch/faiss) for fast similarity search
- [HuggingFace](https://huggingface.co/) for open-source embeddings
- [Groq](https://groq.com/) for high-speed LLM inference
- [Streamlit](https://streamlit.io/) for rapid UI development

---

## 📜 License

This project is intended for educational and research purposes.

---

**⭐ If you find this project useful, please consider giving it a star!**