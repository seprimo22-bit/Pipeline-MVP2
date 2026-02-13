from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI(title="Campbell Cognitive Pipeline")

templates = Jinja2Templates(directory="templates")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QuestionInput(BaseModel):
    question: str


# ---------------- AI FACT RETRIEVAL ----------------

def get_ai_answer(question):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Provide factual, concise information only. "
                        "No opinions. No analysis. Just facts."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI error: {str(e)}"


# ---------------- HOME ROUTE ----------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# ---------------- ANALYZE ROUTE ----------------

@app.post("/analyze")
async def analyze_question(data: QuestionInput):

    ai_answer = get_ai_answer(data.question)

    # Return ONLY facts
    return JSONResponse({
        "facts": ai_answer
    })
