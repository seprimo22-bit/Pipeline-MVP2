from flask import Flask, request, jsonify, render_template
import os
from openai import OpenAI
from rag_engine import search_documents

app = Flask(__name__)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question"}), 400

    # STEP 1 — Ask OpenAI
    try:
        ai_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Give factual, neutral answers only."},
                {"role": "user", "content": question}
            ]
        )

        answer = ai_response.choices[0].message.content

    except Exception as e:
        return jsonify({"error": f"OpenAI error: {str(e)}"}), 500


    # STEP 2 — Check your private docs
    docs = search_documents(question)

    return jsonify({
        "answer": answer,
        "document_support": docs if docs else None,
        "confidence": "document_supported" if docs else "baseline"
    })


if __name__ == "__main__":
    app.run(debug=True)
