from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ------------------------------
# 1️⃣ DOMAIN CLASSIFICATION GATE
# ------------------------------

def is_private_question(question: str) -> bool:
    private_keywords = [
        "paper zero",
        "third harmonic",
        "titan alloy",
        "g-code",
        "campbell",
        "your book",
        "your paper",
        "your theory",
        "orr",
        "constraint-first"
    ]

    q = question.lower()
    return any(keyword in q for keyword in private_keywords)


# ------------------------------
# 2️⃣ GENERAL KNOWLEDGE RESPONSE
# ------------------------------

def generate_general_answer(question: str):

    # Replace this with your actual LLM call
    # This is placeholder for structure

    return {
        "source": "general_knowledge",
        "confidence": "high",
        "document_support": None,
        "answer": (
            "This response was generated using general knowledge only. "
            "Private documents were not searched because the question did not reference them."
        )
    }


# ------------------------------
# 3️⃣ PRIVATE DOCUMENT RESPONSE
# ------------------------------

def generate_private_answer(question: str):

    # This is where your vector search + document retrieval logic goes
    # DO NOT automatically attach raw document text
    # Only summarize relevant portions

    return {
        "source": "private_documents",
        "confidence": "document_based",
        "document_support": "Relevant internal documents consulted.",
        "answer": (
            "This response references your private document corpus. "
            "Specific documents were searched because the question explicitly relates to your work."
        )
    }


# ------------------------------
# 4️⃣ MAIN ASK ROUTE
# ------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
        <h2>Campbell Cognitive Pipeline</h2>
        <form action="/ask" method="post">
            <input name="question" style="width:400px;">
            <button type="submit">Ask</button>
        </form>
    """)


@app.route("/ask", methods=["POST"])
def ask():

    question = request.form.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # 🧠 GATE DECISION
    if is_private_question(question):
        result = generate_private_answer(question)
    else:
        result = generate_general_answer(question)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
