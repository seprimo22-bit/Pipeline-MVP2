from flask import Flask, request, jsonify, render_template
import os
from openai import OpenAI

app = Flask(__name__)

# -----------------------------
# OPENAI SETUP
# -----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# STEP 1 — FACT EXTRACTION
# -----------------------------
def extract_article_facts(article_text):

    prompt = f"""
Extract ONLY factual statements from this article.

Rules:
- No opinions
- No interpretation
- No summary
- Just measurable or stated facts

ARTICLE:
{article_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content


# -----------------------------
# STEP 2 — META ANALYSIS
# -----------------------------
def analyze_article(facts):

    prompt = f"""
Analyze this set of extracted facts about a scientific article.

Do NOT repeat the article.
Do NOT summarize.

Provide factual analysis about:

- Type of publication
- Strength of evidence
- Novelty level
- Presence of experimental validation
- Risk of bias
- Real-world applicability
- Limitations visible from facts

FACTS:
{facts}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content


# -----------------------------
# ROUTES
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()

    if not data or "article" not in data:
        return jsonify({"error": "No article provided"}), 400

    article = data["article"]

    # STEP 1
    facts = extract_article_facts(article)

    # STEP 2
    meta = analyze_article(facts)

    return jsonify({
        "extracted_facts": facts,
        "analysis_about_article": meta
    })


if __name__ == "__main__":
    app.run(debug=True)
