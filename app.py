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

    combined_input = question + "\n" + article

    doc_context = search_docs(combined_input) if combined_input.strip() else ""

    system_prompt = """
You are a constraint-based factual analysis engine.

You MUST structure output in exactly these sections:

FACTS FROM INPUT:
- Explicit statements found in the question or article.

FACTS ABOUT INPUT:
- Independently verifiable domain facts.
- Do not restate the article.

DOCUMENT MATCHES:
- Only direct factual matches from indexed documents.
- If none, say: No direct document match found.

RELATIONSHIP STATUS:
- Direct relationship confirmed
- Possible but unverified overlap
- No relationship found

Rules:
- Do NOT speculate.
- Do NOT suggest future applications.
- Do NOT use narrative language.
- If no connection exists, explicitly say so.
"""

    user_prompt = f"""
QUESTION:
{question}

ARTICLE:
{article}

DOCUMENT CONTEXT:
{doc_context}
"""

    if not combined_input.strip():
        return "Enter a question, article, or both."

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content


@app.route("/", methods=["GET", "POST"])
def index():

    result = ""

    if request.method == "POST":
        q = request.form.get("question")
        a = request.form.get("article")
        result = run_pipeline(q, a)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True),
