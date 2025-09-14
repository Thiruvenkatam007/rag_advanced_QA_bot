import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from rag_pipeline import (
    load_documents, get_embeddings, get_splitters, get_vectorstore,
    get_retriever, get_llm, get_qa_chain
)

# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📄 Document Chatbot (RAG + Mistral)")

# -----------------------------
# Configuration
# -----------------------------
persist_dir = "./chroma_store"
init_mode = "auto"
pdf_path = "/home/thiru/draft-oasis-e1-manual-04-28-2024_edited.pdf"
st.info(f"Using document: {pdf_path}")

# -----------------------------
# Initialize RAG pipeline
# -----------------------------
if "qa_chain" not in st.session_state:
    with st.spinner("Initializing pipeline..."):
        embeddings = get_embeddings()
        parent_splitter, child_splitter = get_splitters()
        vectorstore = get_vectorstore(persist_dir, embeddings)

        docs = None
        if init_mode in ["build", "auto"]:
            docs = load_documents(pdf_path)

        retriever = get_retriever(vectorstore, parent_splitter, child_splitter, persist_dir, docs)
        llm = get_llm()
        st.session_state.qa_chain = get_qa_chain(llm, retriever)
        st.success("Pipeline ready!")

# -----------------------------
# Chat interface
# -----------------------------
if "qa_chain" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("Ask a question about the document:")

    if user_input:
        qa_chain = st.session_state.qa_chain
        with st.spinner("Getting answer..."):
            result = qa_chain({"query": user_input})
            answer = result['result']
            source_docs = result['source_documents']

            # Save conversation
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "bot", "content": answer})

    # Display chat messages
    if st.session_state.messages:
        messages = st.session_state.messages
        user_msg = messages[-2] if len(messages) >= 2 else None
        bot_msg = messages[-1]

        if user_msg and user_msg["role"] == "user":
            st.markdown(f"**You:** {user_msg['content']}")
        if bot_msg and bot_msg["role"] == "bot":
            st.markdown(f"**Bot:** {bot_msg['content']}")

    with st.expander("Show Source Chunks Used(Truncated)"):
        if "source_docs" in locals():
            for i, src in enumerate(source_docs, 1):
                st.markdown(f"**Chunk {i}:** {src.page_content[:1000]}...")
                st.markdown("---")
