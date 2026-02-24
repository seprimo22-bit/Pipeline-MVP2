from flask import Flask, request, jsonify, render_template
import re
import os
import numpy as np
import faiss

app = Flask(__name__)

# =========================================================
# GLOBALS (LAZY LOADED)
# =========================================================

rag = None
pipeline = None

# =========================================================
# RAG ENGINE
# =========================================================

class SimpleRAG:

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.documents = []

    def build_index(self, documents):
        embeddings = self.model.encode(documents)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))
        self.documents = documents

    def retrieve(self, query, top_k=3):
        if self.index is None:
            return []

        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)

        return [
            self.documents[i]
            for i in indices[0]
            if i < len(self.documents)
        ]


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


# =========================================================
# LAZY INITIALIZER
# =========================================================

def initialize_engines():
    global rag, pipeline

    if pipeline is None:
        pipeline = CognitivePipeline()

    if rag is None:
        rag = SimpleRAG()

        docs = [
            "Titan A16 is a constraint-first alloy framework.",
            "The coherence ratio measures structural integrity divided by deformation noise.",
            "ORR enforces validation through falsifiable constraint testing.",
            "Geometry-driven metamaterials modulate mechanical properties.",
            "Additive manufacturing introduces microstructural brittleness."
        ]

        rag.build_index(docs)


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    initialize_engines()

    data = request.json
    article = data.get("article", "")
    context = data.get("context", [])

    result = pipeline.run(article, context)
    result["retrieved_context"] = rag.retrieve(article)

    return jsonify(result)


# =========================================================
# START SERVER (Render Friendly)
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
