

from fastapi import FastAPI
from pydantic import BaseModel
import re
import uvicorn

app = FastAPI(title="Campbell Cognitive Pipeline")


class QuestionInput(BaseModel):
    question: str


def paper_zero_layer(text):
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
    observations = data["facts"]
    rectified = list(set(observations))
    gaps = data["unknowns"]

    return {
        "observations": rectified,
        "gaps": gaps
    }


def orns_stabilization(data):
    return {
        "normalized_observations": data["observations"],
        "uncertainty_flags": data["gaps"]
    }


def axiom_extraction(data):
    axioms = []

    for obs in data["normalized_observations"]:
        if "always" in obs.lower() or "never" in obs.lower():
            axioms.append(obs)

    return {"axioms": axioms}


def extended_decision_framework(data):
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
    verified = list(set(all_data["normalized_observations"]))
    return {"verified_interpretation": verified}


def output_classification(data):
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


@app.get("/")
def home():
    return {"status": "Campbell Cognitive Pipeline Running"}


@app.post("/analyze")
def analyze_question(data: QuestionInput):
    return run_pipeline(data.question)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
