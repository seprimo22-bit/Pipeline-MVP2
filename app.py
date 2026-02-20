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

if __name__ == "__main__":
    app.run(debug=True)
