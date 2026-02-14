import os
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

# IMPORTANT: this pulls your document search
from rag_engine import search_documents

# ---------------------------
# CONFIG
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)


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
                    "content":
                    "Answer using broad scientific, academic, and general knowledge first. "
                    "If document context is provided later, integrate it cautiously."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI error: {str(e)}"


# ---------------------------
# HYBRID PIPELINE
# AI FIRST → DOCUMENTS SECOND
# ---------------------------
def hybrid_pipeline(question):

    # 1️⃣ AI general knowledge first
    general = general_ai_answer(question)

    # 2️⃣ Search your documents
    rag_results = search_documents(question)

    # 3️⃣ Mix results if relevant
    if rag_results:

        doc_context = "\n\n".join(
            f"Source: {r['file']}\n{r['snippet'][:800]}"
            for r in rag_results[:2]
        )

        final_answer = f"""
GENERAL AI KNOWLEDGE:
{general}

------------------------------
DOCUMENT CROSS-CHECK:
{doc_context}

Confidence: MIXED (AI + Documents)
"""
        confidence = "MIXED"

    else:
        final_answer = general
        confidence = "AI ONLY"

    return final_answer, confidence


# ---------------------------
# ROUTES
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():

    data = request.json
    question = data.get("question", "")

    answer, confidence = hybrid_pipeline(question)

    return jsonify({
        "answer": answer,
        "confidence": confidence
    })


# ---------------------------
# RUN SERVER
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
