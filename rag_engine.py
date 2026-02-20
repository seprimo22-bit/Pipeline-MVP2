import os
import faiss
import numpy as np
from openai import OpenAI

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_FILE = "doc_index.faiss"
DOC_STORE = "doc_store.npy"

# Load document index if it exists
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    docs = np.load(DOC_STORE, allow_pickle=True).tolist()
else:
    index = None
    docs = []


# -------------------------
# Embedding helper
# -------------------------
def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")


# -------------------------
# Retrieve relevant docs
# -------------------------
def retrieve_documents(query, k=3):
    if index is None or not query:
        return []

    qvec = embed(query).reshape(1, -1)
    D, I = index.search(qvec, k)

    matches = []
    for i in I[0]:
        if i < len(docs):
            matches.append(docs[i])

    return matches


# -------------------------
# Main extraction function
# -------------------------
def extract_article_facts(article_text, question=None):

    # Retrieve internal docs FIRST
    retrieved_docs = retrieve_documents(question or article_text)

    internal_context = "\n\n".join(retrieved_docs)

    if not article_text:
        article_text = "None provided."

    prompt = f"""
You are a FACT EXTRACTION ENGINE.

RULES:
1. Extract facts explicitly stated.
2. Do not speculate.
3. Internal documents are PRIORITY context.
4. Article text is secondary evidence.

INTERNAL DOCUMENT CONTEXT:
{internal_context}

ARTICLE TEXT:
{article_text}

USER QUESTION:
{question}

Return bullet points only:
- Established Scientific Facts
- Article-Specific Facts
- Unknowns / Limits
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "Fact extraction only."},
                {"role": "user", "content": prompt},
            ],
        )

        return {
            "raw_output": response.choices[0].message.content,
            "document_matches": retrieved_docs
        }

    except Exception as e:
        return {
            "error": str(e),
            "document_matches": []
        }


# -------------------------
# Pipeline wrapper
# -------------------------
def run_fact_pipeline(article_text, question=None):

    facts = extract_article_facts(article_text, question)

    return {
        "status": "success",
        "pipeline": "fact_plus_document_analysis",
        "result": facts,
    }
