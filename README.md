# Campbell Cognitive App Pipeline
# Backend API (FastAPI) — Render-ready
# Author: Cognitive Pipeline Prototype

from fastapi import FastAPI
from pydantic import BaseModel
import re
import uvicorn

app = FastAPI(title="Campbell Cognitive App Pipeline")


# -----------------------------
# DATA MODEL
# -----------------------------

class QuestionInput(BaseModel):
    question: str


# -----------------------------
# PIPELINE CORE FUNCTIONS
# -----------------------------

def paper_zero_layer(text):
    """Separate facts, assumptions, unknowns."""
    sentences = re.split(r'[.!?]', text)

    facts = []
    assumptions = []
    unknowns = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if any(word in s.lower() for word in
               ["maybe", "probably", "guess", "think", "assume"]):
            assumptions.append(s)

        elif any(word in s.lower() for word in
                 ["unknown", "unsure", "not sure", "unclear"]):
            unknowns.append(s)

        else:
            facts.append(s)

    return {
        "facts": facts,
        "assumptions": assumptions,
        "unknowns": unknowns
    }


def orr_core(data):
    """Observe → Rectify → Review."""

    observations = data["facts"]

    # Rectify contradictions simple pass
    rectified = list(set(observations))

    # Review gaps
    gaps = data["unknowns"]

    return {
        "observations": rectified,
        "gaps": gaps
    }


def orns_stabilization(data):
    """Normalize interpretation."""

    stable_interpretation = {
        "normalized_observations": data["observations"],
        "uncertainty_flags": data["gaps"]
    }

    return stable_interpretation


def axiom_extraction(data):
    """Extract recurring constraints."""

    axioms = []

    for obs in data["normalized_observations"]:
        if "always" in obs.lower() or "never" in obs.lower():
            axioms.append(obs)

    return {"axioms": axioms}


def extended_decision_framework(data):
    """Risk / ethics / structural evaluation."""

    risk_flags = []
    ethical_flags = []

    for obs in data["normalized_observations"]:
        if any(w in obs.lower() for w in ["danger", "risk", "harm"]):
            risk_flags.append(obs)

        if any(w in obs.lower() for w in ["ethical", "moral"]):
            ethical_flags.append(obs)

    return {
        "risk_flags": risk_flags,
        "ethical_flags": ethical_flags
    }


def final_orr_pass(all_data):
    """Final assumption cleanup."""

    verified = list(set(all_data["normalized_observations"]))

    return {"verified_interpretation": verified}


def output_classification(data):
    """Final structured classification."""

    facts = data["verified_interpretation"]

    hypotheses = []
    speculation = []
    questions = []

    for f in facts:
        if "could" in f.lower() or "might" in f.lower():
            hypotheses.append(f)

        if "imagine" in f.lower():
            speculation.append(f)

        if "?" in f:
            questions.append(f)

    return {
        "facts": facts,
        "hypotheses": hypotheses,
        "speculation": speculation,
        "questions": questions
    }


# -----------------------------
# FULL PIPELINE EXECUTION
# -----------------------------

def run_pipeline(text):

    pz = paper_zero_layer(text)
    orr = orr_core(pz)
    orns = orns_stabilization(orr)
    axioms = axiom_extraction(orns)
    extended = extended_decision_framework(orns)
    final = final_orr_pass(orns)
    classified = output_classification(final)

    return {
        "paper_zero": pz,
        "orr": orr,
        "orns": orns,
        "axioms": axioms,
        "decision_framework": extended,
        "final_output": classified
    }


# -----------------------------
# API ENDPOINTS
# -----------------------------

@app.get("/")
def home():
    return {"status": "Campbell Cognitive Pipeline Running"}


@app.post("/analyze")
def analyze_question(data: QuestionInput):
    result = run_pipeline(data.question)
    return result


# -----------------------------
# LOCAL DEV RUNNER
# -----------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
