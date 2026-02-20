from flask import Flask, render_template, request, jsonify
from rag_engine import run_fact_pipeline

app = Flask(__name__)


# -------------------------
# Web Interface Route
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        article_text = request.form.get("article_text", "")
        question = request.form.get("question", "")

        result = run_fact_pipeline(article_text, question)

    return render_template("index.html", result=result)


# -------------------------
# JSON API Route (Optional)
# -------------------------
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()

        article_text = data.get("article", "")
        question = data.get("question", "")

        result = run_fact_pipeline(article_text, question)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


# -------------------------
# App Launch
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
