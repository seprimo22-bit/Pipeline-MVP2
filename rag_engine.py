
import os
import faiss
import numpy as np
from openai import OpenAI
import PyPDF2

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DOC_FOLDER = "documents"
INDEX_FILE = "doc_index.faiss"
DOC_STORE = "doc_store.npy"


# -------------------------
# Extract text from documents
# -------------------------
def extract_text_from_file(path):
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
# Embeddings
# -------------------------
def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text[:8000]
    )
    return np.array(response.data[0].embedding, dtype="float32")


# -------------------------
# Build document index
# -------------------------
def build_index():

    docs = []
    vectors = []

    if not os.path.exists(DOC_FOLDER):
        return None, []

    for file in os.listdir(DOC_FOLDER):
        path = os.path.join(DOC_FOLDER, file)
        text = extract_text_from_file(path)

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

    return index, docs


# Load or build index
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    docs = np.load(DOC_STORE, allow_pickle=True).tolist()
else:
    index, docs = build_index()


# -------------------------
# Retrieve internal docs LAST
# -------------------------
def retrieve_documents(query, k=3):
    if index is None:
        return []

    qvec = embed(query).reshape(1, -1)
    D, I = index.search(qvec, k)

    matches = []
    for i in I[0]:
        if i < len(docs):
            matches.append(docs[i][:2000])

    return matches


# -------------------------
# Main pipeline
# -------------------------
def extract_article_facts(article_text, question=None):

    # External science first
    external_context = article_text or "No article provided."

    # Internal docs second
    retrieved_docs = retrieve_documents(question or "")
    internal_context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are a scientific fact extraction engine.

Evidence priority:

1. Published science / article text FIRST
2. Internal documents LAST (treat as speculative)
3. Never present internal docs as established fact.

Separate output into:

- Established Scientific Facts
- Article-Specific Findings
- Internal Hypotheses / Speculative Notes
- Unknowns / Limits

ARTICLE / SCIENCE:
{external_context}

INTERNAL DOCUMENTS (speculative):
{internal_context}

QUESTION:
{question}

Return bullet points only.
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
        "raw_output": response.choices[0].message.content,
        "document_matches": retrieved_docs
    }


def run_fact_pipeline(article_text, question=None):

    facts = extract_article_facts(article_text, question)

    return {
        "status": "success",
        "pipeline": "science_first_documents_last",
        "result": facts,
    }
