import os
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from rag_engine import search_documents

app = Flask(__name__)

# Load OpenAI key safely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------------------
# FACT EXTRACTION PROMPT
# ----------------------------

def build_prompt(question, article_text=None):

    instructions = """
You are a scientific fact extraction engine.

Strict rules:
- Extract measurable, verifiable, testable facts only.
- No summaries.
- No opinions.
- No speculation.
- No narrative language.
- Return structured output under these headings:

1. Established Scientific Facts
2. Article-Supported Facts
3. Unknowns or Unverified Claims

Respond in plain text.
"""

    if article_text:
        return f"""{instructions}

QUESTION:
{question}

ARTICLE TEXT:
{article_text[:10000]}
"""
    else:
        return f"""{instructions}

QUESTION:
{question}
"""


# ----------------------------
# ROUTES
# ----------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():

    try:
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({"error": "No question provided"}), 400

        question = data["question"]
        article = data.get("article")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": build_prompt(question, article)}
            ],
            temperature=0
        )

        ai_output = response.choices[0].message.content

        private_hits = search_documents(question)

        return jsonify({
            "fact_analysis": ai_output,
            "private_document_matches": private_hits
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
