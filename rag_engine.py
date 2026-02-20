import os
import numpy as np
from flask import Flask, request, render_template_string
from openai import OpenAI

# ------------------------------------
# CONFIG
# ------------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DOC_FOLDER = "docs"  # put your reference docs here
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

app = Flask(__name__)

# ------------------------------------
# SIMPLE DOCUMENT RAG INDEX
# ------------------------------------

doc_texts = []
doc_vectors = []

def load_docs():
    global doc_texts, doc_vectors

    if not os.path.exists(DOC_FOLDER):
        return

    for f in os.listdir(DOC_FOLDER):
        path = os.path.join(DOC_FOLDER, f)
        if not f.endswith(".txt"):
            continue

        text = open(path, "r", encoding="utf-8").read()
        emb = client.embeddings.create(
            model=EMBED_MODEL,
            input=text[:8000]
        ).data[0].embedding

        doc_texts.append(text)
        doc_vectors.append(np.array(emb))

load_docs()


def search_docs(query):
    if not doc_vectors:
        return ""

    q_emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    ).data[0].embedding

    q_vec = np.array(q_emb)

    sims = [np.dot(q_vec, v) / (np.linalg.norm(v) * np.linalg.norm(q_vec))
            for v in doc_vectors]

    best_idx = int(np.argmax(sims))
    return doc_texts[best_idx][:3000]


# ------------------------------------
# CORE PIPELINE
# ------------------------------------

def run_pipeline(question=None, article=None):
    question = (question or "").strip()
    article = (article or "").strip()

    doc_context = search_docs(question) if question else ""

    system_prompt = """
You are a factual analysis engine.

Rules:
- Return factual information only.
- No filler.
- Distinguish known facts vs inference.
- When analyzing articles, evaluate claims
  against general knowledge.
"""

    if question and not article:
        user_prompt = f"""
QUESTION:
{question}

REFERENCE DOC CONTEXT:
{doc_context}

Return factual information relevant to the question.
"""

    elif article and not question:
        user_prompt = f"""
ARTICLE:
{article}

Analyze the article objectively.
Return facts ABOUT the claims, not just a summary.
"""

    elif question and article:
        user_prompt = f"""
QUESTION:
{question}

ARTICLE:
{article}

REFERENCE DOC CONTEXT:
{doc_context}

Analyze both together.
Extract facts, evaluate claims, and connect them.
"""

    else:
        return "Enter a question, an article, or both."

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return resp.choices[0].message.content


# ------------------------------------
# UI + ROUTE
# ------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    result = ""

    if request.method == "POST":
        q = request.form.get("question")
        a = request.form.get("article")
        result = run_pipeline(q, a)

    return render_template_string("""
<html>
<head>
<title>Campbell Cognitive Pipeline</title>
<style>
body {font-family: Arial; margin:40px;}
textarea {width:100%; height:140px;}
.result {background:#eee; padding:15px; margin-top:20px;}
button {padding:10px 20px;}
</style>
</head>
<body>

<h1>Campbell Cognitive Pipeline</h1>

<form method="POST">

<h3>Ask a Question</h3>
<textarea name="question"
placeholder="Ask something factual..."></textarea>

<h3>Paste Article (Optional)</h3>
<textarea name="article"
placeholder="Paste article text for analysis..."></textarea>

<br><br>
<button type="submit">Run Analysis</button>

</form>

<div class="result">
<pre>{{result}}</pre>
</div>

</body>
</html>
""", result=result)


if __name__ == "__main__":
    app.run(debug=True)
