from fastapi import FastAPI
from pydantic import BaseModel
import re
import os
from openai import OpenAI
from fastapi.responses import HTMLResponse

# -----------------------
# INITIALIZE APP
# -----------------------

app = FastAPI(title="Campbell Cognitive Pipeline")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------
# INPUT MODEL
# -----------------------

class QuestionInput(BaseModel):
    question: str


# -----------------------
# AI FACT RETRIEVAL
# -----------------------

def get_ai_answer(question):

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content":
                    """Give factual, neutral information.
                    No hype.
                    Separate facts clearly.
                    Keep concise."""
                },
                {"role": "user", "content": question}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI retrieval error: {str(e)}"


# -----------------------
# PAPER ZERO (FACT FILTER)
# -----------------------

def extract_facts(text):

    sentences = re.split(r'[.!?]', text)
    facts = []

    for s in sentences:
        s = s.strip()

        if not s:
            continue

        if any(word in s.lower() for word in
               ["maybe", "possibly", "might", "guess", "speculative"]):
            continue

        facts.append(s)

    return facts


# -----------------------
# PIPELINE (SIMPLIFIED CLEAN)
# -----------------------

def run_pipeline(question):

    ai_answer = get_ai_answer(question)

    facts = extract_facts(ai_answer)

    return {
        "question": question,
        "facts": facts,
        "answer": ai_answer
    }


# -----------------------
# HOME PAGE UI
# -----------------------

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<html>
<head>
<title>Campbell Cognitive Pipeline</title>
<style>
body {
    font-family: Arial;
    max-width: 900px;
    margin: auto;
    padding: 20px;
    background: #0f172a;
    color: #e2e8f0;
}
textarea {
    width: 100%;
    height: 120px;
    background: #020617;
    color: white;
    border: 1px solid #334155;
    padding: 10px;
}
button {
    padding: 10px 20px;
    margin-top: 10px;
    background: #22c55e;
    border: none;
    color: white;
    font-weight: bold;
    cursor: pointer;
}
pre {
    background: #020617;
    padding: 15px;
    margin-top: 20px;
    border: 1px solid #334155;
}
</style>
</head>

<body>

<h2>Campbell Cognitive Pipeline</h2>

<textarea id="q" placeholder="Ask a question..."></textarea>
<br>
<button onclick="ask()">Analyze</button>

<pre id="r">Waiting for question...</pre>

<script>
async function ask() {

    let q = document.getElementById("q").value;

    let res = await fetch("/analyze", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question: q})
    });

    let data = await res.json();

    document.getElementById("r").textContent =
        "FACTS:\\n\\n- " +
        data.facts.join("\\n- ") +
        "\\n\\nANSWER:\\n\\n" +
        data.answer;
}
</script>

</body>
</html>
"""


# -----------------------
# ANALYZE ROUTE
# ----------------
