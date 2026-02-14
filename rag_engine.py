==============================

Campbell Cognitive Pipeline

Hybrid RAG + General Knowledge Fallback

Drop-in replacement example

==============================

import os from flask import Flask, request, jsonify, render_template from openai import OpenAI from rag_engine import search_documents

app = Flask(name) client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

------------------------------

Helper: Ask general model

------------------------------

def general_model_answer(question): response = client.chat.completions.create( model="gpt-4o-mini", messages=[ {"role": "system", "content": "Answer clearly and factually. Label this GENERAL KNOWLEDGE."}, {"role": "user", "content": question} ] ) return response.choices[0].message.content

------------------------------

Main RAG + Fallback Logic

------------------------------

def hybrid_answer(question): rag_result = search_documents(question)

if rag_result and rag_result.get("confidence", 0) > 0.6:
    return {
        "answer": rag_result["answer"],
        "source": "DOCUMENT-RAG",
        "confidence": rag_result["confidence"]
    }

# fallback to general model
general = general_model_answer(question)

return {
    "answer": general,
    "source": "GENERAL-MODEL",
    "confidence": 0.5
}

------------------------------

Flask Routes

------------------------------

@app.route("/") def home(): return render_template("index.html")

@app.route("/ask",
