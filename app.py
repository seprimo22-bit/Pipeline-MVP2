import os
from flask import Flask, render_template, request
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_article_metadata(article_text, question):
    """
    Pulls facts ABOUT the article:
    - What type of study it is
    - Methodology
    - Scope
    - Assumptions
    - Limits
    - Evidence base
    """

    prompt = f"""
You are a strict fact extraction engine.

IMPORTANT RULES:
- Extract facts ABOUT the article, not facts from the article topic.
- Do NOT answer the user's question.
- Do NOT diagnose or speculate.
- Only describe the article itself.

Return structured facts:

1. Article Type
2. Research Goal
3. Methodology Used
4. Evidence/Data Sources
5. Assumptions or Constraints
6. Limits of Conclusions
7. Confidence Level (High/Medium/Low)
8. Unknowns or Missing Information

User Question (context only):
{question}

Article Text:
{article_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        question = request.form.get("question", "")
        article_text = request.form.get("article_text", "")

        if article_text.strip():
            result = analyze_article_metadata(article_text, question)
        else:
            result = "No article text provided."

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
