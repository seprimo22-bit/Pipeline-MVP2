import os
import faiss
import numpy as np
import PyPDF2
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DOC_FOLDER = "documents"

# Global storage
index = None
docs = []


# -------------------------
# TEXT EXTRACTION
# -------------------------
def extract_text(path):
    text = ""
    try:
        if path.lower().endswith(".pdf"):
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        elif path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception as e:
        print("Document read error:", e)
    return text


# -------------------------
# EMBEDDING
# -------------------------
def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text[:8000]
    )
    return np.array(response.data[0].embedding, dtype="float32")


# -------------------------
# BUILD INDEX EVERY STARTUP
# -------------------------
def build_index():
    global index, docs

    print("Building document index...")

    docs = []
    vectors = []

    if not os.path.exists(DOC_FOLDER):
        print("Documents folder not found.")
        index = None
        return

    for file in os.listdir(DOC_FOLDER):
        path = os.path.join(DOC_FOLDER, file)
        text = extract_text(path)

        if len(text) > 100:
            docs.append(text)
            vectors.append(embed(text))

    if not vectors:
        print("No documents indexed.")
        index = None
        return

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))

    print(f"Indexed {len(docs)} documents.")


# Build on import
build_index()


# -------------------------
# RETRIEVE DOCS
# -------------------------
def retrieve_docs(query, k=3):
    if index is None or not query:
        return []

    qvec = embed(query).reshape(1, -1)
    D, I = index.search(qvec, k)

    results = []
    for i in I[0]:
        if i < len(docs):
            results.append(docs[i][:2000])

    return results


# -------------------------
# MAIN PIPELINE
# -------------------------
def run_fact_pipeline(article_text, question=None):

    retrieved_docs = retrieve_docs(question or "")

    prompt = f"""
You are a scientific fact extraction engine.

EVIDENCE ORDER:
1. Established science first.
2. Article text second.
3. Internal documents last (treat as unverified).

Never treat internal documents as confirmed fact.

QUESTION:
{question}

ARTICLE:
{article_text}

INTERNAL DOCUMENTS (UNVERIFIED):
{chr(10).join(retrieved_docs)}

Return:

• Established Scientific Facts
• Article Findings
• Unverified Internal Notes
• Unknowns / Limits

Bullet points only.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "Scientific fact extraction only."},
            {"role": "user", "content": prompt},
        ],
    )

    return {
        "analysis": response.choices[0].message.content,
        "documents_used": len(retrieved_docs)
    }
