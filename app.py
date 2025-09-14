import streamlit as st
from rag_pipeline import build_pipeline

# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="RAG QA bot", page_icon="🤖")
st.title("📄 Document QA bot")

with st.expander("ℹ️ About this Chatbot"):
    st.markdown("""
### 🔍 How it Works
This chatbot is built with **LangChain** and **Mistral AI** for Retrieval-Augmented Generation (RAG).  
It allows you to query a PDF document directly, including **scanned images** inside the PDF thanks to RapidOCR.

---

#### 📄 Document Loading
- Uses **PDFMinerLoader** with `RapidOCRBlobParser` to extract both text and images from PDFs.  
- Scanned PDFs are supported: text inside **images is converted to text** via OCR.  
- Each page is treated as a standalone document before splitting.

---

#### ✂️ Chunking Strategy
- Documents are not passed directly into embeddings. Instead, they are **split into smaller pieces** to optimize retrieval and LLM input.  
- **Two-level splitting (Hierarchical):**  
  - **Parent Chunks (≈2000 characters):** Capture **larger semantic units** like sections or paragraphs.  
  - **Child Chunks (≈400 characters):** Capture **smaller units** like sentences or sub-paragraphs.  

Why this matters:
- Parent chunks ensure **context coherence** (you don’t lose meaning by cutting too small).  
- Child chunks ensure **fine-grained search precision** (retrieval can zoom into very specific details).  
- When answering, the retriever links back to **parent context** so answers don’t become fragmented.

---

#### 📦 Vector Store & Retrieval
- Uses **ChromaDB** to store embeddings persistently.  
- Each chunk is embedded with **HuggingFace MiniLM** (`all-MiniLM-L6-v2`).  

**Retrieval pipeline:**
1. **ParentDocumentRetriever:**  
   - Maps child chunks → parent chunks.  
   - When a child chunk matches the query, the parent chunk is returned, keeping context intact.  
2. **ContextualCompressionRetriever:**  
   - Wraps the base retriever.  
   - Uses **FlashrankRerank** to rerank and filter results, keeping only the most relevant chunks.  
   - Prevents irrelevant or overly long context from being passed to the LLM.  

Benefits:
- More **accurate retrieval** (less noise, better grounding).  
- Ensures **answers are detailed but contextually correct**.  
- Handles large PDFs gracefully by not flooding the LLM with unnecessary text.  

---

#### 🧠 Embeddings
- Uses **dense vector embeddings** (MiniLM, 384 dimensions).  
- Embeddings capture **semantic meaning** (not just keywords).  
- This enables **semantic search**:  
  - Query: *"How to troubleshoot error code 504?"*  
  - Retrieval finds chunks with phrases like *"Resolving 504 gateway timeout issues..."* even if exact words differ.  

---

#### 🤖 LLM
- Powered by **Mistral Small** (`ChatMistralAI`) with temperature=0.7.  
- Prompted for **detailed, grounded responses**.  
- If the context does not support an answer, the LLM avoids hallucinating and instead uses “context not found”.

---

#### 💬 QA Chain
- Built with **RetrievalQA** (`chain_type="stuff"`).  
- Combines multiple retrieved chunks into a single context window.  
- Uses a **custom prompt** tuned for **medical/technical detail**.  

---

✅ **Extra Features**  
- OCR ensures **image-based PDFs** (scans, diagrams with text) are fully searchable.  
- Reranking ensures only the **top ~10 most relevant chunks** are passed, avoiding LLM context overflow.  
- Every answer shows **source chunks** so you can trace back the exact document evidence.
""")

# -----------------------------
# Configuration
# -----------------------------
persist_dir = "./chroma_store"
init_mode = "auto"
pdf_path = "./oasis_manual.pdf"
st.info(f"Using document: {pdf_path}")

# -----------------------------
# Chat interface
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Ask a question about the document:")

if user_input:
    with st.spinner("Getting answer..."):
        result = build_pipeline(pdf_path, persist_dir, user_input, init_mode)
        answer = result['result']
        source_docs = result['source_documents']

        # Save conversation
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "bot", "content": answer})

# -----------------------------
# Show only latest Q&A
# -----------------------------
if st.session_state.messages:
    messages = st.session_state.messages
    user_msg = messages[-2] if len(messages) >= 2 else None
    bot_msg = messages[-1]

    if user_msg and user_msg["role"] == "user":
        st.markdown(f"**You:** {user_msg['content']}")
    if bot_msg and bot_msg["role"] == "bot":
        st.markdown(f"**Bot:** {bot_msg['content']}")

# -----------------------------
# Source chunks expander
# -----------------------------
with st.expander("Show Source Chunks Used (Truncated)"):
    if "source_docs" in locals():
        for i, src in enumerate(source_docs, 1):
            st.markdown(f"**Chunk {i}:** {src.page_content[:1000]}...")
            st.markdown("---")
