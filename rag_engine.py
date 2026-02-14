import os
import glob
from pathlib import Path

# Optional PDF support
try:
    import PyPDF2
except:
    PyPDF2 = None


DOCUMENT_FOLDER = "documents"


# -------------------------------
# TEXT EXTRACTION
# -------------------------------

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

    return docs


DOCUMENT_CACHE = load_documents()


# -------------------------------
# SIMPLE SEARCH FUNCTION
# -------------------------------

def search_documents(question):

    question = question.lower()
    results = []

    for doc in DOCUMENT_CACHE:

        text = doc["content"].lower()

        score = sum(
            1 for word in question.split()
            if word in text
        )

        if score > 0:
            results.append({
                "file": doc["file"],
                "score": score,
                "snippet": doc["content"][:1500]
            })

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:3]
