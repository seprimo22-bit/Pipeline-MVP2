from flask import Flask, render_template, request
from rag_engine import run_fact_pipeline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        article_text = request.form.get("article_text")
        question = request.form.get("question")

        result = run_fact_pipeline(article_text, question)

    return render_template("index.html", result=result)
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
            "private_document_matches": result.get("document_matches", [])
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
