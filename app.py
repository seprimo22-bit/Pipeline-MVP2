from flask import Flask, request, jsonify, render_template
import os
from rag_engine import search_documents

app = Flask(__name__)


# ---------------------------
# DOMAIN GATE
# ---------------------------
def is_private_question(question: str) -> bool:
    private_keywords = [
        "paper zero",
        "third harmonic",
        "titan alloy",
        "g-code",
        "campbell",
        "orr",
        "constraint-first"
    ]

    q = question.lower()
    return any(k in q for k in private_keywords)


# ---------------------------
# GENERAL KNOWLEDGE RESPONSE
# ---------------------------
def generate_general_answer(question):
    return {
        "source": "general_knowledge",
        "confidence": "baseline",
        "document_support": None,
        "answer": (
            "General factual response mode.\n\n"
            "Private documents were NOT searched because "
            "the question did not reference them."
        )
    }


# ---------------------------
# PRIVATE DOCUMENT RESPONSE
# ---------------------------
def generate_private_answer(question):

    docs = search_documents(question)

    return {
        "source": "private_documents",
        "confidence": "document_supported",
        "document_support": docs,
        "answer": (
            "Private document search triggered.\n\n"
            "Relevant document snippets returned below."
        )
    }


# ---------------------------
# ROUTES
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data["question"]

    if is_private_question(question):
        result = generate_private_answer(question)
    else:
        result = generate_general_answer(question)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
