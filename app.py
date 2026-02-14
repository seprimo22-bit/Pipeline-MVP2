import os
import re
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from rag_engine import search_docs


app = FastAPI(title="Campbell Cognitive Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class QuestionInput(BaseModel):
    question: str


# -------- Fact Verification + ORR Structure --------

def analyze_response(answer, context):

    hallucination_flag = "LOW"

    if not context:
        hallucination_flag = "HIGH"
    elif len(answer) > 3 * len(" ".join(context)):
        hallucination_flag = "MEDIUM"

    return {
        "hallucination_risk": hallucination_flag,
        "context_used": len(context) > 0,
        "constraint_check": "PASS" if context else "NO DATA"
    }


# -------- Main Pipeline --------

def get_ai_answer(question):

    context = search_docs(question)

    context_text = "\n\n".join(context[:3])

    system_prompt = f"""
You are a research assistant.

Answer ONLY using the provided context.
If the answer is not present, say:
"No verified reference found."

Context:
{context_text}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )

    answer = completion.choices[0].message.content.strip()

    verification = analyze_response(answer, context)

    return {
        "answer": answer,
        "citations": context[:2],
        "verification": verification
    }


# -------- Routes --------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
async def ask_pipeline(data: QuestionInput):

    try:
        result = get_ai_answer(data.question)
        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
