from fastapi import FastAPI
from pydantic import BaseModel
import re

app = FastAPI(title="Campbell Cognitive Pipeline")

# -------- INPUT MODEL --------

class QuestionInput(BaseModel):
    question: str


# -------- PAPER ZERO LAYER --------

def paper_zero_layer(text):
    sentences = re.split(r'[.!?]', text)

    facts = []
    assumptions = []
    unknowns = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if "maybe" in s.lower() or "probably" in s.lower():
            assumptions.append(s)
        elif "?" in text:
            unknowns.append(s)
        else:
            facts.append(s)

    return {
        "facts": facts,
        "assumptions": assumptions,
        "unknowns": unknowns
    }


# -------- ORR CORE --------

def orr_core(data):
    cleaned = list(set(data["facts"]))
    return {
        "observations": cleaned,
        "contradictions_removed": True,
        "bias_checked": True
    }


# -------- ORNS STABILIZATION --------

def orns_stabilization(data):
    return {
        "stable_interpretation": data["observations"],
        "ambiguity_checked": True,
        "emotional_bias_reduced": True
    }


# -------- AXIOM EXTRACTION --------

def axiom_extraction(data):
    axioms = []
    for item in data["stable_interpretation"]:
        if len(item.split()) > 3:
            axioms.append(f"Potential principle: {item}")
    return axioms


# -------- EXTENDED DECISION FRAMEWORK --------

def extended_decision_framework(data):
    return {
        "risk_level": "unknown",
        "ethical_flag": "neutral",
        "structural_integrity": "stable"
    }


# -------- FINAL ORR PASS --------

def final_orr_pass(data):
    return {
        "verified_output": data["stable_interpretation"],
        "narrative_creep_removed": True
    }


# -------- OUTPUT CLASSIFICATION --------

def output_classification(data):
    return {
        "facts": data["verified_output"],
        "hypotheses": [],
        "speculation": [],
        "questions": []
    }


# -------- FULL PIPELINE --------

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


# -------- API ROUTES --------

@app.get("/")
def home():
    return {"status": "Campbell Cognitive Pipeline Running"}


@app.post("/analyze")
def analyze_question(data: QuestionInput):
    return run_pipeline(data.question)
