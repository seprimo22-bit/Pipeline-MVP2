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
# Extract text
# -------------------------
def extract_text_from_file(path):
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
        print("Doc read error:", e)

    return text


# -------------------------
# Chunk documents (IMPORTANT)
# -------------------------
def chunk_text(text, size=800):

    words = text.split()
    chunks = []

    for i in range(0, len(words), size):
        chunks.append(" ".join(words[i:i+size]))

    return chunks


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
# ALWAYS rebuild index
# -------------------------
def build_index():

    docs = []
    vectors = []

    print("Rebuilding document index...")

    if not os.path.exists(DOC_FOLDER):
        return None, []

    for file in os.listdir(DOC_FOLDER):
        path = os.path.join(DOC_FOLDER, file)
        text = extract_text_from_file(path)

        for chunk in chunk_text(text):
            if len(chunk) > 50:
                docs.append(chunk)
                vectors.append
