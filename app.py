import os
import re
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


# -------- INITIALIZE APP --------
app = FastAPI(title="Campbell Cognitive Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------- REQUEST MODEL --------
class QuestionInput(BaseModel):
    question: str


# -------- AI FACT RETRIEVAL --------
def get_ai_answer(question):

    SYSTEM_PROMPT = """
Provide 12–15 independent factual statements.

Rules:
- Each fact stands alone.
- No speculation or opinion.
- Avoid narrative explanation.
- Short factual sentences.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )

    answer = completion.choices[0].message.content

    # Optional cleanup
    answer = re.sub(r"\n{2,}", "\n", answer.strip())

    return answer


# -------- WEB ROUTES --------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/ask")
async def ask_pipeline(data: QuestionInput):
    try:
        answer = get_ai_answer(data.question)
        return JSONResponse({"answer": answer})

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )
