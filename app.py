from flask import Flask, render_template, request, jsonify
from rag_engine import run_fact_pipeline

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        article_text = request.form.get("article_text", "")
        question = request.form.get("question", "")
        result = run_fact_pipeline(article_text, question)

    return render_template("index.html", result=result)


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    result = run_fact_pipeline(
        article_text=data.get("article", ""),
        question=data.get("question", "")
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
