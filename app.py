from flask import Flask, request, jsonify, render_template
import os
import re
import datetime
from openai import OpenAI

# PDF and DOCX parsing
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

app = Flask(__name__)

# -------------------------------------------------
# OPENAI SETUP
# -------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# -------------------------------------------------
# CORPUS FOLDER — your research docs live here
# User uploads are NEVER saved here
# -------------------------------------------------
CORPUS_FOLDER = "documents"
os.makedirs(CORPUS_FOLDER, exist_ok=True)


# -------------------------------------------------
# TEXT EXTRACTION — handles TXT, PDF, DOCX
# -------------------------------------------------
def extract_text_from_file(filepath):
    """Extract plain text from .txt, .pdf, or .docx files."""
    text = ""

    if filepath.endswith(".txt") or filepath.endswith(".md"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    elif filepath.endswith(".pdf"):
        if PDF_SUPPORT:
            try:
                with pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                text = f"[PDF read error: {str(e)}]"
        else:
            text = "[PDF support not installed — add pdfplumber to requirements.txt]"

    elif filepath.endswith(".docx"):
        if DOCX_SUPPORT:
            try:
                doc = DocxDocument(filepath)
                for para in doc.paragraphs:
                    if para.text.strip():
                        text += para.text + "\n"
            except Exception as e:
                text = f"[DOCX read error: {str(e)}]"
        else:
            text = "[DOCX support not installed — add python-docx to requirements.txt]"

    return text


# -------------------------------------------------
# FACT EXTRACTION
# -------------------------------------------------
def extract_facts(text):
    """Pull factual sentences from input text."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    facts = []
    trigger_words = [
        "is", "was", "were", "has", "have", "had",
        "according", "study", "data", "%",
        "shows", "demonstrates", "indicates", "reveals",
        "defined", "measured", "computed", "found"
    ]
    for s in sentences:
        s = s.strip()
        if len(s) > 20 and any(w in s.lower() for w in trigger_words):
            facts.append(s)
    return facts[:10]  # Cap at 10


# -------------------------------------------------
# CORPUS SEARCH
# Reads TXT, PDF, DOCX from /documents folder
# Matches sentences with 2+ overlapping words
# -------------------------------------------------
def search_corpus(query):
    """Search your research documents for relevant sentences."""
    matches = []

    if not os.path.exists(CORPUS_FOLDER):
        return matches

    # Clean query — remove common stop words for better matching
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on",
        "at", "to", "for", "of", "with", "is", "was", "are",
        "were", "be", "been", "what", "how", "why", "which",
        "that", "this", "it", "its", "by", "from", "as"
    }
    query_words = set(query.lower().split()) - stop_words

    supported = (".txt", ".md", ".pdf", ".docx")

    for filename in os.listdir(CORPUS_FOLDER):
        if not filename.endswith(supported):
            continue

        filepath = os.path.join(CORPUS_FOLDER, filename)
        text = extract_text_from_file(filepath)

        if not text or text.startswith("["):
            continue

        sentences = re.split(r'(?<=[.!?]) +', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            sentence_words = set(sentence.lower().split()) - stop_words
            overlap = query_words & sentence_words
            if len(overlap) >= 3:  # Require 3+ meaningful words
                matches.append({
                    "source": filename,
                    "match": sentence,
                    "overlap_count": len(overlap)
                })

    # Sort by overlap strength, return top 5
    matches.sort(key=lambda x: x["overlap_count"], reverse=True)
    return matches[:5]


# -------------------------------------------------
# HOME
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -------------------------------------------------
# MAIN ANALYZE ROUTE
# Accepts: question, article (pasted OR uploaded text)
# User content is NEVER written to disk
# -------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()
    question = data.get("question", "").strip()
    article = data.get("article", "").strip()

    # Handle base64 uploads (PDF/DOCX sent from browser)
    if article.startswith("data:"):
        try:
            import base64
            header, encoded = article.split(",", 1)
            decoded_bytes = base64.b64decode(encoded)
            article = decoded_bytes.decode("utf-8", errors="ignore")
        except Exception:
            article = ""

    combined_input = f"{question}\n\n{article}".strip() if article else question

    # -------- FACTS FROM INPUT TEXT --------
    input_facts = extract_facts(article) if article else []

    # -------- AI FACTS + CORPUS RELATIONSHIP --------
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

            # Check relationship to Campbell research corpus
            rel_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You check whether input relates to these specific research areas: "
                            "zero baseline mathematics, coherence functionals, "
                            "Kuramoto-Sivashinsky equation analysis, Navier-Stokes coherence, "
                            "constraint triangulation theory, ORR methodology, "
                            "Titan alloy materials science, CTT feasible arena concept, "
                            "Einstein Reversal geometric lattice, Campbell Axiom Ledger, "
                            "ORR observe rectify review pipeline, Paper Zero methodology. "
                            "Reply with ONLY one of these three options exactly: "
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
