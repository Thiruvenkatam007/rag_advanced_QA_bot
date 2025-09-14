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