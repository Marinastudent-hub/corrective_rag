# Corrective RAG System

An intelligent document-based Question Answering system built with **Corrective Retrieval-Augmented Generation (CRAG)**. It retrieves relevant information from uploaded documents, evaluates the retrieved context, and improves retrieval when the results are weak or irrelevant — helping generate more reliable, context-grounded answers.

## 📌 Project Overview

Traditional RAG systems retrieve document chunks and pass them directly to an LLM. However, retrieved information may sometimes be irrelevant or insufficient.

This project addresses that limitation through a corrective retrieval pipeline that evaluates retrieved chunks, rewrites queries when necessary, and generates answers only from relevant document context.

## ✨ Key Features

- **Multi-format Document Support:** Upload PDF, DOCX, TXT, and Markdown files.
- **Document Processing:** Extract, clean, split, and embed document content.
- **Vector Database:** Store and retrieve document chunks using ChromaDB.
- **Retrieval Evaluation:** Use an LLM to assess the relevance of retrieved chunks.
- **Query Rewriting:** Improve search queries when retrieved context is insufficient.
- **Iterative Retrieval:** Retry retrieval with rewritten queries, up to a configurable limit.
- **Context-Grounded Answers:** Generate answers using relevant retrieved information.
- **Source Attribution:** Display document names and page references.
- **Conversation History:** Maintain chat sessions across app refreshes.
- **No-Answer Handling:** Return `NO_ANSWER` when the documents do not contain sufficient information.

## 🏗️ System Architecture

The system follows a corrective RAG pipeline:

![Corrective RAG Architecture](<Corrective Rag.png>)

**Pipeline:**

Document Upload → Text Extraction → Cleaning → Chunking → Embedding → ChromaDB → Retrieval → Relevance Grading → Query Rewriting (if needed) → Re-retrieval → Answer Generation → Answer with Sources

## 🖥️ Application Screenshots

### Document Upload & Processing

![Document Upload](<Screenshot 2026-11-15 220047.png>)

### Question Answering Interface

![Question Answering](<Screenshot 2026-11-15 220100.png>)

## ⚙️ How It Works

1. **Document Ingestion:** The user uploads documents, which are extracted and divided into smaller chunks.
2. **Embedding & Storage:** Text chunks are converted into vector embeddings and stored in ChromaDB.
3. **Initial Retrieval:** The system retrieves the most relevant chunks for the user's question.
4. **Relevance Grading:** An LLM evaluates the retrieved chunks and identifies relevant context.
5. **Query Correction:** If the retrieved context is insufficient, the system rewrites the query and performs another retrieval attempt.
6. **Answer Generation:** The LLM generates an answer using the verified context.
7. **Source Display:** The answer is accompanied by document source references.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web interface |
| ChromaDB | Vector database |
| Sentence Transformers | Text embeddings |
| BAAI/bge-base-en-v1.5 | Embedding model |
| Groq API | LLM inference |
| PyPDF | PDF text extraction |
| python-docx | DOCX text extraction |

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Marinastudent-hub/corrective_rag.git
cd corrective_rag
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory:

```env
LLM_API_KEY=your_groq_api_key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
```

Replace `your_groq_api_key` with your own API key.

**Important:** Never upload your `.env` file or expose your API key publicly.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

## 📁 Project Structure

```text
corrective_rag/
│
├── app.py              # Streamlit application interface
├── config.py           # Project configuration
├── crag.py             # Corrective RAG pipeline
├── ingestion.py        # Document processing and indexing
├── llm.py              # LLM integration
├── requirements.txt    # Project dependencies
├── .gitignore
└── README.md
```

## 🎯 Project Objective

The main objective of this project is to improve the reliability of document-based question answering by introducing a corrective retrieval mechanism that evaluates retrieved information before generating responses.

---

