from flask import Flask, request, jsonify, render_template
import os
from openai import OpenAI
from rag_engine import search_documents

app = Flask(__name__)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------
# OPENAI FACTUAL RESPONSE
# ---------------------------
def get_openai_answer(question):

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Provide factual, neutral information only. "
                        "No diagnosis, no speculation, no opinions."
                    )
                },
                {"role": "user", "content": question}
            ],
            temperature=0.2
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"OpenAI error: {str(e)}"


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

    # 1️⃣ OpenAI answer
    answer = get_openai_answer(question)

    # 2️⃣ Check private docs
    doc_support = search_documents(question)

    return jsonify({
        "answer": answer,
        "document_support": doc_support
    })


if __name__ == "__main__":
    app.run(debug=True)
