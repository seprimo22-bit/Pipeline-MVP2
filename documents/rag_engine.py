import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_FILE = "faiss_index.bin"
TEXT_FILE = "text_chunks.npy"


def extract_pdf_text(folder="documents"):
    texts = []

    if not os.path.exists(folder):
        return texts

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, file))
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    texts.append(txt)

    return texts


def embed_texts(texts):
    embeddings = []

    for text in texts:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000]
        )
        embeddings.append(emb.data[0].embedding)

    return np.array(embeddings).astype("float32")


def build_index():
    texts = extract_pdf_text()

    if not texts:
        return None, []

    embeddings = embed_texts(texts)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    np.save(TEXT_FILE, texts)

    return index, texts


def load_index():
    if os.path.exists(INDEX_FILE):
        return (
            faiss.read_index(INDEX_FILE),
            np.load(TEXT_FILE, allow_pickle=True)
        )

    return build_index()


def search_docs(query, k=3):
    index, texts = load_index()

    if index is None:
        return []

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    q = np.array([emb.data[0].embedding]).astype("float32")
    _, indices = index.search(q, k)

    return [texts[i] for i in indices[0]]
