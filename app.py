from flask import Flask, request, jsonify, render_template
import os
import re
import datetime
from openai import OpenAI

app = Flask(__name__)

# -------------------------------------------------
# OPENAI SETUP
# -------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

CORPUS_FOLDER = "corpus"  # YOUR research docs live here — never user uploads
os.makedirs(CORPUS_FOLDER, exist_ok=True)


# -------------------------------------------------
# FACT EXTRACTION
# -------------------------------------------------
def extract_facts(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    facts = []
    for s in sentences:
        if any(word in s.lower() for word in [
            "is", "was", "were", "has", "have",
            "according", "study", "data", "%",
            "shows", "demonstrates", "indicates", "reveals"
        ]):
            facts.append(s.strip())
    return facts


# -------------------------------------------------
# SEARCH YOUR CORPUS DOCUMENTS
# Reads .txt files from /corpus folder
# These are YOUR research papers — never user files
# -------------------------------------------------
def search_corpus(question):
    matches = []

    if not os.path.exists(CORPUS_FOLDER):
        return matches

    question_words = set(question.lower().split())

    for filename in os.listdir(CORPUS_FOLDER):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(CORPUS_FOLDER, filename)

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        sentences = re.split(r'(?<=[.!?]) +', text)

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = question_words & sentence_words
            # Match if 2+ meaningful words overlap
            if len(overlap) >= 2:
                matches.append({
                    "source": filename,
                    "match": sentence.strip()
                })

    return matches[:5]  # Return top 5 matches max


# -------------------------------------------------
# HOME
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -------------------------------------------------
# MAIN ANALYZE ROUTE
# Accepts: question, article (pasted OR file text)
# User documents are NEVER saved to disk
# -------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()
    question = data.get("question", "").strip()
    article = data.get("article", "").strip()

    # Combined input for analysis
    combined_input = question
    if article:
        combined_input = f"{question}\n\n{article}" if question else article

    # -------- FACTS FROM PASTED/UPLOADED TEXT --------
    input_facts = extract_facts(article) if article else []

    # -------- INTERNET / AI FACTS --------
    facts_about_input = []
    relationship_status = "No relationship determined."

    if client and combined_input:
        try:
            # Facts about the topic
            facts_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a fact extractor. "
                            "Return only verifiable factual bullet points about the topic. "
                            "No opinions, no diagnosis, no conclusions. "
                            "Format: one fact per line starting with -"
                        )
                    },
                    {"role": "user", "content": combined_input}
                ],
                max_tokens=400
            )
            facts_text = facts_response.choices[0].message.content
            facts_about_input = [
                line.strip().lstrip("-").strip()
                for line in facts_text.split("\n")
                if line.strip().startswith("-")
            ]

            # Relationship to Campbell corpus
            rel_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You check whether input relates to these research areas: "
                            "zero baseline mathematics, coherence functionals, "
                            "Kuramoto-Sivashinsky equation analysis, Navier-Stokes coherence, "
                            "constraint triangulation theory, ORR methodology, "
                            "Titan alloy materials science, CTT feasible arena concept, "
                            "Einstein Reversal geometric lattice. "
                            "Reply with ONLY one of: "
                            "'Direct relationship confirmed', "
                            "'Possible but unverified overlap', or "
                            "'No relationship found'."
                        )
                    },
                    {"role": "user", "content": combined_input}
                ],
                max_tokens=20
            )
            relationship_status = rel_response.choices[0].message.content.strip()

        except Exception as e:
            facts_about_input = [f"API error: {str(e)}"]

    # -------- CORPUS DOCUMENT SEARCH --------
    corpus_matches = search_corpus(combined_input)

    # -------- RESPONSE --------
    return jsonify({
        "timestamp": str(datetime.datetime.now()),
        "facts_from_input": input_facts,
        "facts_about_input": facts_about_input,
        "corpus_matches": corpus_matches,
        "relationship_status": relationship_status
    })


# -------------------------------------------------
# SERVER
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
