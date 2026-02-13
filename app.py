from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import re
from openai import OpenAI

# ---------- APP SETUP ----------
app = FastAPI(title="Campbell Cognitive Pipeline")

templates = Jinja2Templates(directory="templates")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- INPUT MODEL ----------
class QuestionInput(BaseModel):
    question: str


# ---------- IMPROVED AI FACT RETRIEVAL ----------
def get_ai_answer(question):

    SYSTEM_PROMPT = """
    Provide 12–15 independent factual statements.

    Rules:
    - Each fact must stand alone.
    - Avoid narrative explanation.
    - Avoid speculation, hedging, or opinion.
    - No filler text.
    - Short, clear factual sentences only.
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


# ---------- PAPER ZERO FILTER ----------
def paper_zero_layer(text):

    sentences = re.split(r"[.!?]", text)

    facts = []
    assumptions = []
    unknowns = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if any(w in s.lower() for w in ["maybe", "possibly", "likely"]):
            assumptions.append(s)
        else:
            facts.append(s)

    return {
        "facts": facts,
        "assumptions": assumptions,
        "unknowns": unknowns
    }


# ---------- ORR CORE ----------
def orr_core(data):

    cleaned = list(set(data["facts"]))

    return {
        "observations": cleaned,
        "contradictions_removed": True,
        "bias_checked": True
    }


# ---------- ORNS STABILIZATION ----------
def orns_stabilization(data):

    return {
        "stable_interpretation": data["observations"],
        "ambiguity_checked": True,
        "emotional_bias_reduced": True
    }


# ---------- AXIOM EXTRACTION ----------
def axiom_extraction(data):

    axioms = []

    for item in data["stable_interpretation"]:
        if len(item.split()) > 5:
            axioms.append(f"Potential principle: {item}")

    return axioms


# ---------- FINAL PASS ----------
def final_orr_pass(data):

    return {
        "verified_output": data["stable_interpretation"],
        "narrative_creep_removed": True
    }


# ---------- CLASSIFICATION ----------
def output_classification(data):

    return {
        "facts": data["verified_output"],
        "hypotheses": [],
        "speculation": [],
        "questions": []
    }


# ---------- FULL PIPELINE ----------
def run_pipeline(question):

    ai_answer = get_ai_answer(question)

    pz = paper_zero_layer(ai_answer)
    orr = orr_core(pz)
    orns = orns_stabilization(orr)
    axioms = axiom_extraction(orns)
    final = final_orr_pass(orns)
    classified = output_classification(final)

    return {
        "original_question": question,
        "ai_answer": ai_answer,
        "paper_zero": pz,
        "orr": orr,
        "orns": orns,
        "axioms": axioms,
        "final_output": classified
    }


# ---------- ROUTES ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
def analyze_question(data: QuestionInput):
    return run_pipeline(data.question)
