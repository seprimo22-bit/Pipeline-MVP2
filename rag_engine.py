import os
import numpy as np
import PyPDF2
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DOC_FOLDER = "documents"
DOCUMENT_CHUNKS = []


# -------------------------
# Extract text from PDFs/TXT
# -------------------------
def extract_text(path):
    try:
        if path.lower().endswith(".pdf"):
            text = ""
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text

        elif path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

    except Exception as e:
        print("Doc read error:", e)

    return ""


# -------------------------
# Chunk documents
# -------------------------
def chunk_text(text, size=800):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i+size])
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks


# -------------------------
# Load documents ONCE
# -------------------------
def load_documents():
    if not os.path.exists(DOC_FOLDER):
        print("Documents folder missing.")
        return

    for file in os.listdir(DOC_FOLDER):
        path = os.path.join(DOC_FOLDER, file)
        text = extract_text(path)

        if text:
            DOCUMENT_CHUNKS.extend(chunk_text(text))

    print(f"Loaded {len(DOCUMENT_CHUNKS)} document chunks.")


load_documents()


# -------------------------
# Embedding helper
# -------------------------
def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text[:8000]
    )
    return np.array(response.data[0].embedding, dtype="float32")


# -------------------------
# Retrieve relevant docs
# -------------------------
def retrieve_docs(query, k=3):

    if not DOCUMENT_CHUNKS or not query:
        return []

    qvec = embed(query)

    scores = []
    for chunk in DOCUMENT_CHUNKS:
        dvec = embed(chunk)
        similarity = np.dot(qvec, dvec) / (
            np.linalg.norm(qvec) * np.linalg.norm(dvec)
        )
        scores.append((similarity, chunk))

    scores.sort(reverse=True)
    return [s[1] for s in scores[:k]]


# -------------------------
# Main pipeline
# -------------------------
def run_fact_pipeline(article_text="", question=""):

    retrieved_docs = retrieve_docs(question)
    doc_context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are a factual analysis engine.

ORDER OF EVIDENCE:

1. General scientific knowledge FIRST.
2. Article analysis SECOND.
3. Internal documents LAST (treat as speculative notes).

Never present internal documents as established fact.

OUTPUT SECTIONS:

• Established Scientific Facts
• Facts About Article / Topic
• Internal Speculative Notes
• Unknowns / Limits

ARTICLE:
{article_text}

QUESTION:
{question}

INTERNAL DOCUMENT NOTES:
{doc_context}

Bullet points only.
No speculation.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "Fact analysis only."},
            {"role": "user", "content": prompt},
        ],
    )

    return {
        "status": "success",
        "analysis": response.choices[0].message.content,
        "documents_used": len(retrieved_docs)
    }
