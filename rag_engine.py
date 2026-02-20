import os
import glob
from pathlib import Path

try:
    import PyPDF2
except:
    PyPDF2 = None

DOCUMENT_FOLDER = "documents"

# Words that cause garbage matches
STOPWORDS = {
    "the","a","an","is","are","of","to","and","in",
    "on","for","with","this","that","it","as","at",
    "be","by","from","or"
}

# ---------------------------
# TEXT EXTRACTION
# ---------------------------

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
                text += page.extract_text() or ""
    except:
        pass

    return text


# ---------------------------
# LOAD DOCUMENTS ONCE
# ---------------------------

def load_documents():
    docs = []

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

        if content.strip():
            docs.append({
                "file": os.path.basename(path),
                "content": content.lower()
            })

    print(f"Loaded {len(docs)} private documents.")
    return docs


DOCUMENT_CACHE = load_documents()


# ---------------------------
# SMART DOCUMENT SEARCH
# ---------------------------

def search_documents(question):

    if not DOCUMENT_CACHE:
        return []

    # Remove stopwords
    q_words = [
        w.lower() for w in question.split()
        if w.lower() not in STOPWORDS and len(w) > 2
    ]

    results = []

    for doc in DOCUMENT_CACHE:

        score = sum(
            1 for word in q_words
            if word in doc["content"]
        )

        # Require stronger match
        if score >= 3:
            results.append({
                "file": doc["file"],
                "score": score,
                "snippet": "Relevant document identified (content hidden for privacy)."
            })

    # Sort best first
    results.sort(key=lambda x: x["score"], reverse=True)

    # Only return if confident
    if results and results[0]["score"] >= 4:
        return results[:3]

    return []
