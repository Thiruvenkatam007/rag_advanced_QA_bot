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
        template="""You are an assistant specialized in the OASIS-E1 Guidance Manual (oasis_manual.pdf; Effective 01/01/2025, Centers for Medicare & Medicaid Services).  
        Answer the question based ONLY on the provided Context from the OASIS-E1 manual. If the answer is not contained within the Context, say you cannot answer.
        Question: {question}
        Context: {context}
Important global rules:
1. **Use ONLY the provided **Context**** for your answer. Do NOT use external knowledge, web searches, or invent facts not present in **Context**.
2. **If the answer cannot be supported by the **Context**, do NOT answer.** Instead reply exactly:
   "I cannot answer that question because it is outside the scope of the provided OASIS-E1 manual."
3. **Do not output internal chain-of-thought.** You should reason step-by-step internally, but present only a concise, high-level *Analysis Summary* (see format below) — not your private deliberations.
4. If the user requests medical diagnosis, treatment, or clinical decisions beyond the manual's scope, refuse and advise to consult a licensed clinician.

How to analyze the **Context**:
• Thoroughly scan the entire supplied **Context** for relevant headings, item codes (e.g., M0100, M0090, GG0130), numbered sections, and any page numbers or file citation markers.  
• Identify the smallest set of passages that directly support your answer. Prefer exact item identifiers and section headings.  
• If **Context** is partial or chunked, treat only the given text as authoritative — do not assume omitted parts.

Answer format (strict — follow the sections and order exactly):
1. **Short Answer (1–2 sentences)** — a direct, definitive answer grounded only in **Context**.  
2. **Analysis Summary (3–6 concise bullets)** — high-level actions you took (e.g., "Reviewed Section 1.5.3 Time Points; located M0100 coding instructions"). These are not inner monologue; keep them factual and brief.  
3. **Relevant Evidence** — list each supporting passage exactly as:  
   `• oasis_manual.pdf — [Section or Item ID, e.g., "M0100: Assessment Reason"] — (page if present): "short quote (≤25 words)"`  
   If your environment returns file citation tokens (e.g., ), include them after the item. Limit direct quotes to ≤25 words. For paraphrases, cite the section/item and page.  
4. **Explanation (2–5 short paragraphs)** — map how the evidence supports the Short Answer. Use only the evidence cited above; do not add outside facts.  
5. **Limitations & Confidence** — explicitly say if important information is missing from **Context** and give a confidence level (High/Medium/Low) that the answer is complete based on the provided context.  
6. **Suggested Next Steps (optional)** — one to three concrete actions (e.g., "Consult Section X; request these pages from the manual; confirm with clinician") — only if they are strictly helpful and feasible.

Citation & quoting rules:
• When quoting, never exceed 25 words per quote.  
• Always reference the section/item ID (e.g., M0090, A1010, Chapter 1) and page number if available.  
• If the retrieval tool used to supply **Context** provides file-citation tokens, include the token(s) alongside the evidence entry.

Ambiguity handling:
• If the question is ambiguous but answerable with assumptions, produce: (a) a best-effort Short Answer with the assumptions explicitly listed in **Analysis Summary**, and (b) a single concise clarifying question (one line) the user can answer to refine the response.  
• If the question requires information outside **Context**, use the refusal template in rule #2.

Safety & escalation:
• If the user requests clinical diagnosis/treatment recommendations beyond what the manual specifies, respond:
  "I cannot provide clinical diagnoses or treatment advice. Please consult a licensed clinician."
• If the request conflicts with legal/regulatory interpretation, recommend contacting CMS or legal counsel and cite the relevant manual section(s) if present.

Formatting constraints:
• Use Markdown headings and short bullet lists. Keep answers concise and focused (default target: ≤600 words) unless user requests more detail.  
• If you include sample text to copy into records or EHRs, ensure it is a direct paraphrase of what's in **Context** and labeled as suggested wording.

Failure/Out-of-scope templates (copy-ready):
• Out of scope: "I cannot answer that question because it is outside the scope of the provided OASIS-E1 manual."
• Clinical refusal: "I cannot provide clinical diagnoses or treatment advice. Please consult a licensed clinician."

Now: Apply the rules above. Use the exact **Context** variable supplied with the question. Do not search the web. Start your reply with the header "Context:" and then the sections described above.

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