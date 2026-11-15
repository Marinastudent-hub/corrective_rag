"""Corrective RAG: retrieve -> grade -> (rewrite -> re-retrieve) -> generate."""
import config
import llm
from ingestion import embed, get_collection

NO_ANSWER = (
    "I couldn't find an answer to your question in the uploaded documents. "
    "Try rephrasing the question or upload a document that contains this information."
)


# ---------- Retrieve ----------
def retrieve(query: str, k: int = config.TOP_K):
    col = get_collection()
    if col.count() == 0:
        return []
    res = col.query(
        query_embeddings=embed([query]),
        n_results=min(k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": d, "source": m["source"], "page": m["page"], "distance": dist}
        for d, m, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]



# ---------- Grade (evaluate all chunks in one API call) ----------
GRADER_SYSTEM = """
You are a relevance grader for a RAG system.
Evaluate each document chunk independently against the question.

A chunk is relevant if it contains information that directly helps answer
the question, even if it does not provide the complete answer.

Return ONLY valid JSON in this format:
{"relevant_indices": [1, 3]}

Use the original chunk numbers, starting from 1.
Return an empty list if no chunks are relevant.
"""


def grade_chunks(question: str, chunks: list):
    if not chunks:
        return []

    chunks_text = "\n\n".join(
        f"[{i}] {chunk['text']}"
        for i, chunk in enumerate(chunks, 1)
    )

    out = llm.chat_json(
        GRADER_SYSTEM,
        f"Question: {question}\n\nDocument chunks:\n{chunks_text}"
    )

    indices = out.get("relevant_indices", [])

    return [
        chunk
        for i, chunk in enumerate(chunks, 1)
        if i in indices
    ]


# ---------- Rewrite ----------
REWRITE_SYSTEM = (
    "You rewrite user questions to improve document retrieval. "
    "Make the query clearer and more specific, add likely keywords or synonyms, "
    "write the query in English, and do NOT answer it. "
    "Return ONLY the rewritten query."
)


def rewrite_query(question: str, tried: list) -> str:
    return llm.chat(
        REWRITE_SYSTEM,
        f"Original question: {question}\nQueries already tried (did not work): {tried}",
        temperature=0.3,
    )


# ---------- Generate ----------
ANSWER_SYSTEM = (
    "You answer questions using ONLY the provided context. "
    "If the context does not contain the answer, reply exactly: NO_ANSWER. "
    "Never use outside knowledge. Always answer in English. "
    "Cite sources inline using their numbers like [1], [2]."
)


def generate_answer(question: str, chunks: list) -> str:
    context = "\n\n".join(
        f"[{i}] (source: {c['source']}, page {c['page']})\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    return llm.chat(ANSWER_SYSTEM, f"Context:\n{context}\n\nQuestion: {question}")


# ---------- Full pipeline ----------

def answer(question: str) -> dict:
    trace = []
    tried = [question]
    query = question

    for attempt in range(config.MAX_RETRIES + 1):
        chunks = retrieve(query)
        verified = grade_chunks(question, chunks)

        trace.append(
            f"Attempt {attempt + 1}: query = \"{query}\" -> "
            f"{len(chunks)} chunks retrieved, {len(verified)} judged relevant"
        )

        if verified:
            text = generate_answer(question, verified)

            if "NO_ANSWER" not in text:
                return {
                    "answer": text,
                    "sources": verified,
                    "trace": trace,
                    "answered": True,
                }

            trace.append("The LLM decided the context is insufficient")

        else:
            trace.append("No relevant context found")

        if attempt < config.MAX_RETRIES:
            query = rewrite_query(question, tried)
            tried.append(query)
            trace.append("Context is weak -> rewriting the query")

    return {
        "answer": NO_ANSWER,
        "sources": [],
        "trace": trace,
        "answered": False,
    }
