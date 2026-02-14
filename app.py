import os
import numpy as np
import faiss
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI
from PyPDF2 import PdfReader

# ------------------ APP SETUP ------------------

app = FastAPI(title="Campbell Cognitive Pipeline")
templates = Jinja2Templates(directory="templates")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DOC_FOLDER = "documents"
INDEX_FILE = "faiss.index"
META_FILE = "doc_meta.npy"

# ------------------ REQUEST MODEL ------------------

class QuestionInput(BaseModel):
    question: str

# ------------------ DOCUMENT LOADING ------------------

def load_documents():
    texts = []
    meta = []

    for file in os.listdir(DOC_FOLDER):
        if file.endswith(".pdf"):
            path = os.path.join(DOC_FOLDER, file)
            reader = PdfReader(path)

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    texts.append(text)
                    meta.append({"file": file, "page": i+1})

    return texts, meta


# ------------------ EMBEDDINGS ------------------

def embed_texts(texts):
    embeddings = []

    for t in texts:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=t[:3000]
        )
        embeddings.append(emb.data[0].embedding)

    return np.array(embeddings).astype("float32")


# ------------------ BUILD INDEX ------------------

def build_index():
    texts, meta = load_documents()

    if not texts:
        return None, None, None

    emb = embed_texts(texts)

    index = faiss.IndexFlatL2(len(emb[0]))
    index.add(emb)

    faiss.write_index(index, INDEX_FILE)
    np.save(META_FILE, {"texts": texts, "meta": meta})

    return index, texts, meta


def load_index():
    if not os.path.exists(INDEX_FILE):
        return build_index()

    index = faiss.read_index(INDEX_FILE)
    data = np.load(META_FILE, allow_pickle=True).item()

    return index, data["texts"], data["meta"]


# ------------------ SEARCH ------------------

def search_docs(query, k=3):
    index, texts, meta = load_index()

    if index is None:
        return []

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    q = np.array([emb.data[0].embedding]).astype("float32")
    distances, indices = index.search(q, k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "text": texts[idx],
            "file": meta[idx]["file"],
            "page": meta[idx]["page"],
            "distance": float(distances[0][i])
        })

    return results


# ------------------ ANSWER GENERATION ------------------

def generate_answer(question):

    docs = search_docs(question)

    if not docs:
        return {"answer": "No verified reference found."}

    context = "\n\n".join([d["text"][:1500] for d in docs])

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer ONLY using provided context. "
                    "If uncertain, say so. Avoid speculation."
                ),
            },
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ],
    )

    answer = completion.choices[0].message.content

    # -------- CONFIDENCE SCORE --------
    avg_distance = sum(d["distance"] for d in docs) / len(docs)

    if avg_distance < 0.5:
        confidence = "HIGH"
    elif avg_distance < 1.0:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    hallucination_flag = (
        "LOW RAG SUPPORT" if confidence == "LOW" else "SUPPORTED"
    )

    citations = "\n".join(
        [f"{d['file']} (p.{d['page']})" for d in docs]
    )

    structured = f"""
--- CITATIONS ---
{citations}

--- CONFIDENCE ---
{confidence}

--- HALLUCINATION CHECK ---
{hallucination_flag}

--- ORR VALIDATION ---
Constraint Anchor: Document-grounded
Entropy Risk: {confidence}
Verification Status: {hallucination_flag}
"""

    return {"answer": answer + "\n\n" + structured}


# ------------------ ROUTES ------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
async def ask_pipeline(data: QuestionInput):
    result = generate_answer(data.question)
    return JSONResponse(result)
