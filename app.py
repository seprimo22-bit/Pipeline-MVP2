import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# CORS (prevents browser blocking)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Request schema
class QuestionRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
async def ask_pipeline(data: QuestionRequest):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Provide clear factual answers. Avoid speculation."
                },
                {
                    "role": "user",
                    "content": data.question
                }
            ],
        )

        answer = completion.choices[0].message.content

        return JSONResponse({"answer": answer})

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )
