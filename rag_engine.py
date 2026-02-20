import os
import numpy as np
from openai import OpenAI
import PyPDF2

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"
DOC_FOLDER = "documents"

doc_chunks = []
doc_vectors = []


def extract_text(path):
    if path.endswith(".pdf"):
        text = ""
        reader = PyPDF2.PdfReader(path)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        return open(path, encoding="utf-8").read()


def build_index():
    global doc_chunks, doc_vectors

    for f in os.listdir(DOC_FOLDER):
        path = os.path.join(DOC_FOLDER, f)
        if not f.endswith((".txt", ".pdf")):
            continue

        text = extract_text(path)

        chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]

        for chunk in chunks:
            emb = client.embeddings.create(
                model=EMBED_MODEL,
                input=chunk
            ).data[0].embedding

            doc_chunks.append(chunk)
            doc_vectors.append(np.array(emb))


def search_docs(query):
    if not doc_vectors:
        return ""

    q_emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    ).data[0].embedding

    q_vec = np.array(q_emb)

    sims = [np.dot(q_vec, v) /
            (np.linalg.norm(q_vec) * np.linalg.norm(v))
            for v in doc_vectors]

    best = np.argsort(sims)[-3:]
    return "\n\n".join(doc_chunks[i] for i in best)
