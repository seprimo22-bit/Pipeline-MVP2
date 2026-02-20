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
            return f.read().lower()
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
                extracted = page.extract_text()
                if extracted:
                    text += extracted.lower()
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
                "content": content
            })

    print(f"Loaded {len(docs)} private documents.")
    return docs


DOCUMENT_CACHE = load_documents()


# ---------------------------
# DOCUMENT SEARCH
# ---------------------------
def search_documents(question):

    if not DOCUMENT_CACHE:
        return []

    # Clean query words
    q_words = [
        w.lower() for w in question.split()
        if w.lower() not in STOPWORDS and len(w) > 2
    ]

    if not q_words:
        return []

    results = []

    for doc in DOCUMENT_CACHE:
        score = sum(
            1 for word in q_words
            if word in doc["content"]
        )

        # Require at least minimal relevance
        if score >= 2:
            results.append({
                "file": doc["file"],
                "score": score,
                "snippet": "Relevant document identified (content hidden for privacy)."
            })

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:3]
