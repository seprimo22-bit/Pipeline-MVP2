from flask import Flask, request, jsonify, render_template
import re
import os
import PyPDF2

app = Flask(__name__)

# =========================================================
# PIPELINE ENGINE
# =========================================================

TERM_MAP = {
    "metamaterial": ["architected material", "lattice material"],
    "coherence": ["stability", "structural integrity"],
    "geometry optimization": ["design-centered", "architected"]
}

AMBIGUOUS_TERMS = [
    "promising", "may", "potential",
    "suggests", "could", "likely"
]


class CognitivePipeline:

    def extract_facts(self, text):
        sentences = re.split(r'[.!?]', text)
        triggers = [
            "developed", "demonstrates", "shows",
            "improves", "reveals", "indicates",
            "introduces", "confirms"
        ]
        return [
            s.strip()
            for s in sentences
            if any(t in s.lower() for t in triggers)
        ]

    def normalize_terms(self, facts):
        normalized = []
        for fact in facts:
            updated = fact
            for canonical, variants in TERM_MAP.items():
                for variant in variants:
                    updated = re.sub(
                        variant,
                        canonical,
                        updated,
                        flags=re.IGNORECASE
                    )
            normalized.append(updated)
        return normalized

    def detect_relationship(self, facts, context_keywords):
        matches = sum(
            keyword.lower() in fact.lower()
            for fact in facts
            for keyword in context_keywords
        )

        if matches >= 3:
            return "Direct relationship confirmed"
        elif matches >= 1:
            return "Conceptual alignment"
        return "No relationship found"

    def ambiguity_score(self, text):
        count = sum(text.lower().count(t) for t in AMBIGUOUS_TERMS)
        total_words = max(len(text.split()), 1)
        return round(count / total_words, 4)

    def confidence_score(self, facts, relationship):
        base = min(len(facts) * 0.1, 0.5)
        if relationship == "Direct relationship confirmed":
            base += 0.3
        elif relationship == "Conceptual alignment":
            base += 0.15
        return round(min(base, 1.0), 3)

    def run(self, article_text, context_keywords=None):
        facts = self.extract_facts(article_text)
        normalized = self.normalize_terms(facts)
        relationship = self.detect_relationship(normalized, context_keywords or [])
        ambiguity = self.ambiguity_score(article_text)
        confidence = self.confidence_score(facts, relationship)

        return {
            "facts_extracted": facts,
            "normalized_facts": normalized,
            "relationship_status": relationship,
            "ambiguity_score": ambiguity,
            "confidence_score": confidence
        }


pipeline = CognitivePipeline()


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    article = request.form.get("article", "")
    context = request.form.get("context", "")
    file = request.files.get("file")

    # Handle uploaded file
    if file:
        if file.filename.endswith(".pdf"):
            article += "\n" + extract_pdf_text(file)
        else:
            article += "\n" + file.read().decode("utf-8", errors="ignore")

    result = pipeline.run(article, context.split())

    return jsonify(result)


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
