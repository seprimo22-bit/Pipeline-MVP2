import os
import glob
import numpy as np
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# CONFIG
# -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DOCS_PATH = "documents"

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)

# -----------------------
# LOAD DOCUMENTS (RAG)
# -----------------------
documents = []
doc_vectors = []

def load_documents():
    global documents, doc_vectors
    documents.clear()
    doc_vectors.clear()

    for file in glob.glob(f"{DOCS_PATH}/*.pdf"):
        try:
            reader = PdfReader(file)
            text = ""

            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t

            if text.strip():
                documents.append((file, text[:4000]))

                emb = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text[:4000]
                )

                doc_vectors.append(emb.data[0].embedding)

        except Exception as e:
            print("Document load error:", e)

load_documents()

# -----------------------
# GENERAL KNOWLEDGE MODEL
# -----------------------
def general_answer(question):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Provide factual scientific information first."
                },
                {"role": "user", "content": question}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"General model error: {e}"

# -----------------------
# RAG SEARCH
# -----------------------
def rag_search(question):
    if not documents:
        return None, 0

    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    sims = cosine_similarity([q_emb], doc_vectors)[0]
    idx = np.argmax(sims)

    return documents[idx], sims[idx]

# -----------------------
# HYBRID PIPELINE
# -----------------------
def hybrid_pipeline(question):
    general = general_answer(question)
    rag_doc, rag_score = rag_search(question)

    if rag_doc and rag_score > 0.5:
        filename, text = rag_doc
        answer = f"""
GENERAL KNOWLEDGE:
{general}

--- DOCUMENT CONTEXT ---
{text[:800]}

SOURCE: {filename}
"""
        confidence = "MIXED SUPPORT"
    else:
        answer = general
        confidence = "GENERAL KNOWLEDGE"

    return answer, rag_score, confidence

# -----------------------
# ROUTES
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    answer, score, conf = hybrid_pipeline(question)

    return jsonify({
        "answer": answer,
        "rag_score": float(score),
        "confidence": conf
    })

# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
