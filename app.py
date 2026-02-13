import os
import re
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

# ----------------------------
# INIT APP
# ----------------------------

app = FastAPI(title="Campbell Cognitive Pipeline")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# REQUEST MODEL
# ----------------------------

class QuestionInput(BaseModel):
    question: str


# ----------------------------
# PAPER ZERO FILTER
# ----------------------------

def paper_zero_layer(text):
    sentences = re.split(r"[.!?]", text)

    facts = []
    for s in sentences:
        s = s.strip()

        if len(s) > 15:
            facts.append(s)

    return facts


# ----------------------------
# AI FACT RETRIEVAL
# ----------------------------

def get_ai_facts(question):

    SYSTEM_PROMPT = """
Provide 12–15 independent factual statements.

Rules:
- Facts must stand alone.
- No narrative explanation.
- No speculation.
- No hedging.
- Clear concise sentences only.
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0
        )

        raw_text = completion.choices[0].message.content
        return paper_zero_layer(raw_text)

    except Exception as e:
        return [f"AI retrieval error: {str(e)}"]


# ----------------------------
# API ENDPOINT
# ----------------------------

@app.post("/ask")
async def ask_question(data: QuestionInput):
    facts = get_ai_facts(data.question)
    return JSONResponse({"facts": facts})


# ----------------------------
# SIMPLE TEST PAGE
# ----------------------------

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <body>
            <h2>Campbell Cognitive Pipeline</h2>
            <form action="/ask" method="post">
                <input name="question" style="width:300px"/>
                <button type="submit">Ask</button>
            </form>
        </body>
    </html>
    """
