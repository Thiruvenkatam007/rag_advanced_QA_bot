# Advanced RAG QA Bot
<img width="1341" height="597" alt="image" src="https://github.com/user-attachments/assets/1103b5d7-1509-4208-9c1e-4435123cb4e7" />

This is an advanced Retrieval-Augmented Generation (RAG) chatbot designed to interact with PDF documents. It leverages LangChain and Mistral AI to provide accurate and contextually aware answers, even from PDFs containing scanned images.
## 🌐 Live Web App

You can try the live web app at: [https://thiruvenkatam007-rag-advanced-qa-bot-app-q0lfr9.streamlit.app/](https://thiruvenkatam007-rag-advanced-qa-bot-app-q0lfr9.streamlit.app/) (May be in sleep mode reach out to me in case you need to access it)
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

## 🏗️ System Architecture

```
+-------------------+      +------------------+      +----------------------+
|   PDF Document    |----->| PDFMinerLoader & |----->|  RecursiveCharacter  |
| (text and images) |      | RapidOCRParser   |      |    TextSplitter      |
+-------------------+      +------------------+      +----------------------+
                                                           |
                                                           v
+----------------------+      +------------------+      +----------------------+
| HuggingFace          |<-----| ChromaDB         |<-----| ParentDocument       |
| Embeddings           |      | (Vector Store)   |      | Retriever            |
| (all-MiniLM-L6-v2)   |      +------------------+      +----------------------+
+----------------------+                                     |
                                                               v
+----------------------+      +------------------+      +----------------------+
| Mistral Small (LLM)  |<-----| RetrievalQA      |<-----| ContextualCompression|
|                      |      | (Custom Prompt)  |      | Retriever w/         |
|                      |      |                  |      | FlashrankRerank      |
+----------------------+      +------------------+      +----------------------+
        ^                                                        |
        |                                                        v
+-------+--------------------------------------------------------+-------+
|                           User Interface (Streamlit)                   |
+------------------------------------------------------------------------+
```

## ⚙️ Installation

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/your-username/rag_advanced_QA_bot.git
    cd rag_advanced_QA_bot
    ```

2.  **Install dependencies:**
    Ensure you have Python 3.8+.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set your Mistral AI API Key:**
    Create a `.env` file with your Mistral AI API key:
    ```
    MISTRAL_API_KEY="your_mistral_api_key"
    ```

## 🚀 Usage

1.  **Place your PDF:**
    Put the PDF document you want to query in the root of the repository and name it `oasis_manual.pdf`, or update the path in `app.py`.

2.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

3.  **Ask Questions:**
    Open the local URL provided in your browser and start asking questions about the document.

## 📦 Dependencies

- streamlit==1.49.1
- langchain==0.3.27
- langchain-community==0.3.29
- langchain-text-splitters==0.3.9
- langchain-chroma==0.2.6
- langchain-mistralai==0.2.11
- langchain-openai==0.3.33
- chromadb==1.0.21
- FlashRank==0.2.10
- pdfminer.six==20250506
- rapidocr-onnxruntime==1.4.4
- huggingface-hub==0.34.4
- sentence-transformers==5.1.0
- torch==2.8.0
- tqdm==4.67.1
- pydantic==2.11.7
- pysqlite3-binary

## 🔧 Configuration

You can modify the following variables in the `app.py` file:

- `persist_dir`: Directory for the ChromaDB vector store.
- `init_mode`: How to initialize the vector store ("auto", "new", or "existing").
- `pdf_path`: Path to your PDF document.

## 📈 Future Enhancements

- **Support for Multiple Documents**: Add the ability to query multiple PDFs at once.
- **Better UI**: Enhance the UI with more advanced features like conversation history and document highlighting.
- **Integration with Other LLMs**: Add support for other language models, such as GPT-4 or Llama.
- **Metadata Filtering**: Implement the ability to filter document chunks based on metadata for improved retrieval accuracy.
