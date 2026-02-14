import os
import glob
from pathlib import Path
from openai import OpenAI

# Optional PDF support
try:
    import PyPDF2
except:
    PyPDF2 = None

# ------------------------
# CONFIG
# ------------------------
DOCUMENT_FOLDER = "documents"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------
# TEXT EXTRACTION
# ------------------------
def read_text_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""


def read_pdf_file(path):
    if not PyPDF2:
        return ""

    text = ""
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t
    except:
        pass

    return text


# ------------------------
# LOAD DOCUMENTS + EMBEDDINGS
# ------------------------
DOCUMENT_CACHE = []
DOC_EMBEDDINGS = []


def load_documents():
    global DOCUMENT_CACHE, DOC_EMBEDDINGS

    DOCUMENT_CACHE.clear()
    DOC_EMBEDDINGS.clear()

    if not os.path.exists(DOCUMENT_FOLDER):
        print("No documents folder found.")
        return

    for path in glob.glob(f"{DOCUMENT_FOLDER}/**/*", recursive=True):

        if os.path.isdir(path):
            continue

        ext = Path(path).suffix.lower()

        if ext == ".pdf":
            content = read_pdf_file(path)
        elif ext in [".txt", ".md"]:
            content = read_text_file(path)
        else:
            continue

        if not content.strip():
            continue

        content = content[:4000]

        try:
            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=content
            )

            DOCUMENT_CACHE.append({
                "file": os.path.basename(path),
                "content": content
            })

            DOC_EMBEDDINGS.append(emb.data[0].embedding)

        except Exception as e:
            print("Embedding error:", e)


load_documents()


# ------------------------
# SEMANTIC SEARCH
# ------------------------
def search_documents(question, top_k=3):

    if not DOCUMENT_CACHE:
        return []

    try:
        q_emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=question
        ).data[0].embedding

    except Exception as e:
        print("Query embedding error:", e)
        return []

    # cosine similarity manually
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity([q_emb], DOC_EMBEDDINGS)[0]

    ranked = sorted(
        zip(DOCUMENT_CACHE, sims),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    results = []

    for doc, score in ranked:
        results.append({
            "file": doc["file"],
            "score": float(score),
            "snippet": doc["content"][:1200]
        })

    return results
