import os
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from rag_engine import search_documents

# ----------------------------
# CONFIG
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

# ----------------------------
# GENERAL KNOWLEDGE FIRST
# ----------------------------
def general_answer(question):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Provide factual, neutral information first. "
                        "If uncertain, say so clearly."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI error: {e}"


# ----------------------------
# HYBRID PIPELINE
# ----------------------------
def hybrid_pipeline(question):

    # Step 1 — General AI knowledge
    general = general_answer(question)

    # Step 2 — Your document search
    rag_results = search_documents(question)

    if rag_results:
        citations = "\n".join(
            f"- {r['file']} (score {r['score']})"
            for r in rag_results
        )

        answer = f"""
GENERAL KNOWLEDGE:
{general}

--- DOCUMENT SUPPORT ---
{rag_results[0]['snippet'][:800]}

CITATIONS:
{citations}

CONFIDENCE: MIXED SUPPORT
"""
    else:
        answer = f"""
GENERAL KNOWLEDGE:
{general}

(No relevant document support found.)

CONFIDENCE: GENERAL KNOWLEDGE ONLY
"""

    return answer


# ----------------------------
# ROUTES
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")

    answer = hybrid_pipeline(question)

    return jsonify({"answer": answer})


# ----------------------------
# RUN (Render compatible)
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
