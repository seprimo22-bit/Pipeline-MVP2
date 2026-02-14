
import os
import glob
import numpy as np
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# CONFIG
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DOCS_PATH = "documents"

app = Flask(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

documents = []
doc_vectors = []


# ---------------------------
# LOAD DOCUMENTS (RAG LIGHT MODE)
# ---------------------------
def load_documents():
    global documents, doc_vectors

    documents = []
    doc_vectors = []

    for file in glob.glob(f"{DOCS_PATH}/*.pdf"):
        try:
            reader = PdfReader(file)
            text = ""

            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text()

            if text.strip():
                documents.append((file, text[:3000]))

                emb = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text[:3000]
                )

                doc_vectors.append(emb.data[0].embedding)

        except Exception as e:
            print("Doc load error:", e)


load_documents()


# ---------------------------
# GENERAL AI KNOWLEDGE FIRST
# ---------------------------
def general_ai_answer(question):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Provide clear factual answers using general "
                        "scientific and public knowledge."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"General AI error: {e}"


# ---------------------------
# LIGHT DOCUMENT SUPPORT
# ---------------------------
def rag_support(question):
    if not documents:
        return None, 0

    try:
        q_emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=question
        ).data[0].embedding

        sims = cosine_similarity([q_emb], doc_vectors)[0]
        idx = np.argmax(sims)

        return documents[idx], sims[idx]

    except Exception as e:
        print("RAG error:", e)
        return None, 0


# ---------------------------
# HYBRID PIPELINE (AI FIRST)
# ---------------------------
def hybrid_pipeline(question):

    general = general_ai_answer(question)
    rag_doc, rag_score = rag_support(question)

    if rag_doc and rag_score > 0.45:
        filename, text = rag_doc

        answer = f"""
{general}

--- OPTIONAL DOCUMENT CONTEXT (Campbell Research) ---

{text[:600]}

Source: {filename}
"""

        confidence = "HYBRID (AI PRIMARY + DOC SUPPORT)"

    else:
        answer = general
        confidence = "GENERAL AI PRIMARY"

    return answer, float(rag_score), confidence


# ---------------------------
# ROUTES
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")

    answer, score, confidence = hybrid_pipeline(question)

    return jsonify({
        "answer": answer,
        "rag_score": score,
        "confidence": confidence
    })


# ---------------------------
# RUN SERVER
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
