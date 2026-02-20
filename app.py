import os
from flask import Flask, request, jsonify, render_template
from rag_engine import extract_article_facts

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():

    try:
        data = request.get_json()

        question = data.get("question", "")
        article = data.get("article", "")

        result = extract_article_facts(article, question)

        return jsonify({
            "fact_analysis": result.get("raw_output", ""),
            "private_document_matches": []
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
