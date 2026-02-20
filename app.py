import os
from flask import Flask, render_template, request
from openai import OpenAI
from rag_engine import build_index, search_docs

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4o-mini"

app = Flask(__name__)

# Build document index once at startup
build_index()


def run_pipeline(question=None, article=None):

    question = (question or "").strip()
    article = (article or "").strip()

    doc_context = search_docs(question) if question else ""

    system = """
You are a factual analysis engine.

Rules:
- Return factual information only.
- No filler or mode labels.
- Evaluate claims objectively.
- Integrate document evidence when relevant.
"""

    if question and not article:
        user = f"""
QUESTION:
{question}

DOCUMENT CONTEXT:
{doc_context}

Return factual information answering the question.
"""

    elif article and not question:
        user = f"""
ARTICLE:
{article}

Analyze objectively.
Return facts about claims, validity, and context.
"""

    elif question and article:
        user = f"""
QUESTION:
{question}

ARTICLE:
{article}

DOCUMENT CONTEXT:
{doc_context}

Analyze all sources together.
"""

    else:
        return "Enter a question, article, or both."

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )

    return resp.choices[0].message.content


@app.route("/", methods=["GET", "POST"])
def index():

    result = ""

    if request.method == "POST":
        q = request.form.get("question")
        a = request.form.get("article")
        result = run_pipeline(q, a)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
