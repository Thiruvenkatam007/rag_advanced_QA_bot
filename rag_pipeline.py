import os
import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever
from langchain.storage import InMemoryStore, create_kv_docstore, LocalFileStore
from langchain_community.document_compressors import FlashrankRerank
from langchain_mistralai import ChatMistralAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# -----------------------------
# 1. Load documents
# -----------------------------
def load_documents(pdf_path: str):
    loader = PDFMinerLoader(
        pdf_path,
        mode="single",
        images_inner_format="markdown-img",
        images_parser=RapidOCRBlobParser(),
    )
    return loader.load()

# -----------------------------
# 2. Create embeddings & text splitters
# -----------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_splitters():
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    return parent_splitter, child_splitter

# -----------------------------
# 3. Create / Load Vectorstore (persistent)
# -----------------------------
import shutil, tempfile
def get_vectorstore(persist_dir: str, embeddings, clear_existing: bool = True):
    # # Make sure the base directory exists
    os.makedirs(persist_dir, exist_ok=True)

    # if clear_existing and os.path.exists(persist_dir):
    #     shutil.rmtree(persist_dir)
    #     os.makedirs(persist_dir, exist_ok=True)

    # # Create a unique subdirectory for this session
    # n_persist_dir = tempfile.mkdtemp(dir=persist_dir)
    return Chroma(
        collection_name="split_parents",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

# -----------------------------
# 4. Create retriever with compression
# -----------------------------
# def get_retriever(vectorstore, parent_splitter, child_splitter,persist_dir, docs=None):
#     store = InMemoryStore()  
#     retriever = ParentDocumentRetriever(
#         vectorstore=vectorstore,
#         docstore=store,
#         child_splitter=child_splitter,
#         parent_splitter=parent_splitter,
#         search_kwargs={"k": 10},
#     )
#     if docs:
#         retriever.add_documents(docs)
#     compressor = FlashrankRerank()
#     return ContextualCompressionRetriever(
#         base_compressor=compressor,
#         base_retriever=retriever,
#     )

from langchain.storage import LocalFileStore
from langchain.storage import create_kv_docstore

def get_retriever(vectorstore, parent_splitter, child_splitter, persist_dir, docs=None):
    # LocalFileStore handles persistence
    kv_store = LocalFileStore(persist_dir)
    # Wrap it so it can handle Document serialization
    store = create_kv_docstore(kv_store)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 20},
    )
    if docs:
        retriever.add_documents(docs)

    compressor = FlashrankRerank(top_n=10)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever,
    )
# -----------------------------
# 5. Setup LLM
# -----------------------------
def get_llm():
    import streamlit as st
    api_key = st.secrets["api_keys"]["mistral"]
    os.environ["MISTRAL_API_KEY"] = api_key
    return ChatMistralAI(model="mistral-small", temperature=0.7)

# -----------------------------
# 6. Setup QA Chain
# -----------------------------
def get_qa_chain(llm, retriever):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the following context to answer the question in detail (medical domain):

Context:
{context}

Question:
{question}

Answer:
"""
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

# -----------------------------
# 7. Query execution
# -----------------------------
def run_query(qa_chain, query: str):
    result = qa_chain({"query": query})
    print("Answer:", result['result'])
    print("\nSource Chunks Used:")
    for src in result['source_documents']:
        print("-", src.page_content, "...")
        print("*" * 80)
    return result

# -----------------------------
# Pipeline Runner with modes
# -----------------------------
def build_pipeline(pdf_path: str, persist_dir: str, query: str, init_mode: str = "auto"):
    embeddings = get_embeddings()
    parent_splitter, child_splitter = get_splitters()
    vectorstore = get_vectorstore(persist_dir, embeddings)

    retriever = None
    if init_mode == "build":
        docs = load_documents(pdf_path)
        retriever = get_retriever(vectorstore, parent_splitter, child_splitter,persist_dir, docs)
        # vectorstore.persist()
    elif init_mode == "load":
        retriever = get_retriever(vectorstore, parent_splitter, child_splitter, persist_dir)
    elif init_mode == "auto":
        existing = vectorstore.get()
        if not existing["ids"]:
            print("No existing vectorstore found. Building new one...")
            docs = load_documents(pdf_path)
            retriever = get_retriever(vectorstore, parent_splitter, child_splitter,persist_dir, docs)
            # vectorstore.persist()
        else:
            print("Loading existing vectorstore...")
            retriever = get_retriever(vectorstore, parent_splitter, child_splitter,persist_dir)

    llm = get_llm()
    qa_chain = get_qa_chain(llm, retriever)
    return run_query(qa_chain, query)


# # -----------------------------
# # Example Run
# # -----------------------------


# if __name__ == "__main__":
#     pdf_path = "/home/thiru/draft-oasis-e1-manual-04-28-2024_edited.pdf"
#     persist_dir = "./chroma_store"
#     query = "what is this document about?"

#     # Modes: "build", "load", or "auto"
#     results = build_pipeline(pdf_path, persist_dir, query, init_mode="load")
# results