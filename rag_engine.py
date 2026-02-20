import os
import faiss
import numpy as np
import PyPDF2
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DOC_FOLDER = "documents"
INDEX_FILE = "doc_index.faiss"
DOC_STORE = "doc_store.npy"


# -------------------------
# TEXT EXTRACTION
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
        print("Document read error:", e)

    return ""


# -------------------------
# EMBEDDINGS
# -------------------------
def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text[:8000]
    )
    return np.array(response.data[0].embedding, dtype="float32")


# -------------------------
# BUILD INDEX IF NEEDED
# -------------------------
def build_index():

    docs = []
    vectors = []

    if not os.path.exists(DOC_FOLDER):
        print("Documents folder missing.")
        return None, []

    for file in os.listdir(DOC_FOLDER):
        path = os.path.join(DOC_FOLDER, file)
        text = extract_text(path)

        if len(text) > 100:
            docs.append(text)
            vectors.append(embed(text))

    if not vectors:
        return None, []

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))

    faiss.write_index(index, INDEX_FILE)
    np.save(DOC_STORE, docs)

    print("Document index built.")
    return index, docs


# LOAD OR BUILD INDEX
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    docs = np.load(DOC_STORE, allow_pickle=True).tolist()
else:
    index, docs = build_index()


# -------------------------
# DOCUMENT RETRIEVAL
# -------------------------
def retrieve_docs(query, k=3):

    if index is None or not query:
        return []

    qvec = embed(query).reshape(1, -1)
    D, I = index.search(qvec, k)

    matches = []
    for i in I[0]:
        if i < len(docs):
            matches.append(docs[i][:2000])

    return matches


# -------------------------
# MAIN FACT PIPELINE
# -------------------------
def run_fact_pipeline(article_text, question=None):

    retrieved_docs = retrieve_docs(question or "")

    doc_context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are a scientific fact extraction engine.

ORDER OF EVIDENCE:

1. Established scientific knowledge FIRST.
2. Article text SECOND.
3. Internal documents LAST (treat as unverified notes).

Never present internal documents as confirmed fact.

Separate output into:

• Established Scientific Facts
• Article Findings
• Unverified Internal Notes
• Unknowns / Limits

ARTICLE:
{article_text}

QUESTION:
{question}

INTERNAL DOCUMENT NOTES:
{doc_context}

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
        "status": "success",
        "analysis": response.choices[0].message.content,
        "documents_used": len(retrieved_docs)
    }
