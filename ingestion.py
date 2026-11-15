"""Ingestion pipeline: upload -> extract -> clean -> split -> embed -> store."""
import hashlib
import io
import re

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import config

_embedder = None
_collection = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        _collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def embed(texts):
    return get_embedder().encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    ).tolist()


# ---------- 1) Extract ----------
def extract_pages(filename: str, data: bytes):
    """Returns a list of (page_number, text)."""
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    if name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(data))
        return [(1, "\n".join(p.text for p in doc.paragraphs))]
    if name.endswith((".txt", ".md")):
        return [(1, data.decode("utf-8", errors="ignore"))]
    raise ValueError("Unsupported file type (PDF / DOCX / TXT / MD only)")


# ---------- 2) Clean ----------
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)        # words split across lines
    text = re.sub(r"[ \t]+", " ", text)           # extra spaces
    text = re.sub(r"\n{3,}", "\n\n", text)        # too many blank lines
    return text.strip()


# ---------- 3) Split ----------
def split_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = []
    for page_no, text in pages:
        text = clean_text(text)
        if not text:
            continue
        for piece in splitter.split_text(text):
            chunks.append({"text": piece, "page": page_no})
    return chunks


# ---------- 4) Embed + Store ----------
def ingest_file(filename: str, data: bytes) -> int:
    pages = extract_pages(filename, data)
    chunks = split_pages(pages)
    if not chunks:
        raise ValueError("Could not extract any text from the file (it may be a scanned/image PDF).")

    texts = [c["text"] for c in chunks]
    embeddings = embed(texts)
    ids = [
        hashlib.md5(f"{filename}-{c['page']}-{i}-{c['text'][:50]}".encode()).hexdigest()
        for i, c in enumerate(chunks)
    ]
    metadatas = [{"source": filename, "page": c["page"]} for c in chunks]

    get_collection().upsert(
        ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
    )
    return len(chunks)


def list_documents():
    data = get_collection().get(include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"]})


def reset_db():
    global _collection
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    client.delete_collection(config.COLLECTION_NAME)
    _collection = None
