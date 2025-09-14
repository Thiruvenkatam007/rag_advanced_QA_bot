import streamlit as st
from rag_pipeline import build_pipeline
import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="RAG QA bot", page_icon="🤖")
st.title("📄 Document QA bot")

with st.expander("ℹ️ About this Chatbot"):
    st.markdown("""
# Advanced RAG QA Bot

This is an advanced Retrieval-Augmented Generation (RAG) chatbot designed to interact with PDF documents. It leverages LangChain and Mistral AI to provide accurate and contextually aware answers, even from PDFs containing scanned images.

## 🚀 Features

- **Chat with PDFs**: Ask questions about your PDF documents and get direct, relevant answers.
- **OCR Support**: Thanks to RapidOCR, it natively works with scanned documents and images within PDFs.
- **Advanced Retrieval**: Uses a parent-child chunking strategy for precise information retrieval.
- **Re-ranking**: Employs FlashRank to reduce noise and pass only the most relevant context to the LLM.
- **Source Verification**: Includes source document chunks in every answer, allowing you to trace the origin of information.
- **Persistent Vector Store**: Uses ChromaDB for storing embeddings, enabling fast access and reusability.

## 🔧 How it Works

This chatbot follows a sophisticated RAG pipeline to ensure answers are accurate and grounded in the document's content.

### 📄 Document Loading

- **PDFMinerLoader**: Used to extract text and images from PDF files.
- **RapidOCRBlobParser**: Used to perform Optical Character Recognition (OCR) on images within the PDF, making scanned documents fully searchable.

### ✂️ Chunking Strategy

Documents are not passed directly into embeddings. Instead, they are split into smaller pieces to optimize retrieval and LLM input.

- **Two-level Splitting (Hierarchical)**:
    - **Parent Chunks (≈2000 characters)**: Capture larger semantic units like sections or paragraphs.
    - **Child Chunks (≈400 characters)**: Capture smaller units like sentences or sub-paragraphs.

This strategy ensures retrieval is precise while maintaining contextual coherence for answers.

### 📦 Vector Store and Retrieval

- **ChromaDB**: Used to store embeddings persistently.
- **HuggingFace MiniLM**: Each chunk is embedded using the `all-MiniLM-L6-v2` model.

**Retrieval Pipeline:**

1.  **ParentDocumentRetriever**: Maps child chunks to their parent chunks. When a child chunk matches a query, the full parent chunk is returned, preserving context.
2.  **ContextualCompressionRetriever**: Wraps the base retriever and uses **FlashrankRerank** to re-rank and filter results, ensuring only the most relevant chunks are passed to the LLM.

### 🧠 Embeddings

- **Dense Vector Embeddings**: Using MiniLM (384 dimensions), the system captures semantic meaning of text beyond simple keyword matching. This allows it to understand and answer questions that don't use the exact words in the document.

### 🤖 LLM

- **Mistral Small**: The `ChatMistralAI` model (with temperature=0.7) is used to generate detailed, grounded responses. If the retrieved context doesn't contain an answer, the LLM is prompted to avoid hallucination and instead respond with "context not found".

### 💬 QA Chain

- **RetrievalQA**: With `chain_type="stuff"`, it combines multiple retrieved chunks into a single context window.
- **Custom Prompt**: Uses a custom prompt tuned for medical and technical details, ensuring answers are accurate and contextually appropriate.

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
