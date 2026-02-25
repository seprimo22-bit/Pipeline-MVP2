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

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -------------------------------------------------
# SIMPLE FACT EXTRACTION (ORIGINAL STYLE)
# -------------------------------------------------
def extract_facts(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    facts = []

    for s in sentences:
        if any(word in s.lower() for word in [
            "is", "was", "were", "has", "have",
            "according", "study", "data", "%"
        ]):
            facts.append(s.strip())

    return facts


# -------------------------------------------------
# SEARCH USER DOCUMENTS
# -------------------------------------------------
def search_local_docs(question):
    matches = []

    if not os.path.exists(UPLOAD_FOLDER):
        return matches

    for file in os.listdir(UPLOAD_FOLDER):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(UPLOAD_FOLDER, file)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        facts = extract_facts(text)

        for fact in facts:
            if any(word in fact.lower() for word in question.lower().split()):
                matches.append({
                    "source": file,
                    "fact": fact
                })

    return matches


# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -------------------------------------------------
# ANALYZE QUESTION (RESTORED ORIGINAL BEHAVIOR)
# -------------------------------------------------
@app.route("/analyze_question", methods=["POST"])
def analyze_question():

    # KEEP ORIGINAL JSON INPUT FORMAT
    data = request.get_json()
    question = data.get("question", "")

    # -------- INTERNET FACTS --------
    if not client:
        internet_facts = "OpenAI API key missing."
    else:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Provide factual concise information only. No diagnosis or opinions."
                    },
                    {"role": "user", "content": question}
                ]
            )

            internet_facts = response.choices[0].message.content

        except Exception as e:
            internet_facts = f"API error: {str(e)}"

    # -------- LOCAL DOCUMENT FACTS --------
    local_matches = search_local_docs(question)

    return jsonify({
        "timestamp": str(datetime.datetime.now()),
        "question": question,
        "internet_facts": internet_facts,
        "local_document_matches": local_matches
    })


# -------------------------------------------------
# DOCUMENT UPLOAD ROUTE (NEW BUT SAFE)
# -------------------------------------------------
@app.route("/upload_doc", methods=["POST"])
def upload_doc():

    file = request.files.get("file")

    if not file:
        return {"error": "No file uploaded"}, 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    return {
        "status": "uploaded",
        "file": file.filename
    }


# -------------------------------------------------
# SERVER START
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
