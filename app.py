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

def extract_article_facts(article_text, question=None):

    if not article_text or len(article_text.strip()) < 50:
        return "No article text provided or text too short."

    prompt = f"""
You are a FACT EXTRACTION ENGINE.

STRICT RULES:
1. ONLY extract facts explicitly stated in the article text.
2. DO NOT infer or answer the question.
3. The question is context only.
4. Separate into:
   - Established Scientific Facts
   - Article-Specific Facts
   - Unknowns / Limits

QUESTION:
{question if question else "None"}

ARTICLE:
{article_text}

Return clean bullet points only.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "Fact extraction only."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        article_text = request.form.get("article_text")
        question = request.form.get("question")

        result = extract_article_facts(article_text, question)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
