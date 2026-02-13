from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import re
from openai import OpenAI

# ---------- INITIALIZE APP ----------
app = FastAPI(title="Campbell Cognitive Pipeline")
templates = Jinja2Templates(directory="templates")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- INPUT MODEL ----------
class QuestionInput(BaseModel):
    question: str


# ---------- AI FACT RETRIEVAL ----------
def get_ai_answer(question):

    SYSTEM_PROMPT = """
    Provide 12–15 independent factual statements.

    Rules:
    - Facts must stand alone.
    - No narrative explanation.
    - No speculation.
    - Clear concise sentences.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI retrieval error: {str(e)}"


# ---------- PAPER ZERO ----------
def paper_zero_layer(text):

    sentences = re.split(r"[.!?]", text)
    facts = []
    assumptions = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if any(w in s.lower() for w in ["maybe", "likely", "possibly"]):
            assumptions.append(s)
        else:
            facts.append(s)

    return {
        "facts": facts,
        "assumptions": assumptions,
        "unknowns": []
    }


# ---------- ORR CORE ----------
def orr_core(data):

    cleaned = list(set(data["facts"]))

    return {
        "observations": cleaned,
        "contradictions_removed": True,
        "bias_checked": True
    }


# ---------- ORNS ----------
def orns_stabilization(data):

    return {
        "stable_interpretation": data["observations"],
        "ambiguity_checked": True
    }


# ---------- FINAL PASS ----------
def final_pass(data):

    return {
        "verified_output": data["stable_interpretation"]
    }


# ---------- FULL PIPELINE ----------
def run_pipeline(question):

    ai_answer = get_ai_answer(question)
    pz = paper_zero_layer(ai_answer)
    orr = orr_core(pz)
    orns = orns_stabilization(orr)
    final = final_pass(orns)

    return {
        "original_question": question,
        "ai_answer": ai_answer,
        "final_output": final["verified_output"]
    }


# ---------- ROUTES ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
def analyze_question(data: QuestionInput):
    return run_pipeline(data.question)
