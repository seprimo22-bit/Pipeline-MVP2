from flask import Flask, request, jsonify import re

app = Flask(name)

-----------------------------

TERM NORMALIZATION MAP

-----------------------------

TERM_MAP = { "metamaterial": ["architected material", "lattice material"], "coherence": ["stability", "structural integrity"], "geometry optimization": ["design-centered", "architected"] }

AMBIGUOUS_TERMS = [ "promising", "may", "potential", "suggests", "could", "likely" ]

-----------------------------

COGNITIVE PIPELINE CLASS

-----------------------------

class CognitivePipeline: def extract_facts(self, text): sentences = re.split(r'[.!?]', text) return [ s.strip() for s in sentences if any(word in s.lower() for word in [ "developed", "demonstrates", "shows", "improves", "reveals", "indicates" ]) ]

def normalize_terms(self, facts):
    normalized = []
    for fact in facts:
        for canonical, variants in TERM_MAP.items():
            for v in variants:
                fact = fact.replace(v, canonical)
        normalized.append(fact)
    return normalized

def detect_relationships(self, facts, context):
    if not context:
        return "No relationship"

    matches = 0
    for fact in facts:
        for keyword in context:
            if keyword.lower() in fact.lower():
                matches += 1

    if matches >= 3:
        return "Direct relationship confirmed"
    elif matches >= 1:
        return "Conceptual alignment"
    else:
        return "No relationship found"

def ambiguity_score(self, text):
    score = sum(text.lower().count(t) for t in AMBIGUOUS_TERMS)
    return round(score / max(len(text.split()), 1), 3)

def compute_confidence(self, facts, relationship):
    base = len(facts) * 0.1

    if relationship == "Direct relationship confirmed":
        base += 0.3
    elif relationship == "Conceptual alignment":
        base += 0.15

    return min(base, 1.0)

def run(self, article_text, query_context=None):
    facts = self.extract_facts(article_text)
    normalized = self.normalize_terms(facts)
    relationship = self.detect_relationships(normalized, query_context)
    ambiguity = self.ambiguity_score(article_text)
    confidence = self.compute_confidence(facts, relationship)

    return {
        "facts": facts,
        "normalized_terms": normalized,
        "relationship_status": relationship,
        "ambiguity_score": ambiguity,
        "confidence_score": confidence
    }

pipeline = CognitivePipeline()

-----------------------------

API ENDPOINT

-----------------------------

@app.route('/analyze', methods=['POST']) def analyze(): data = request.json article = data.get('article', '') context = data.get('context', [])

result = pipeline.run(article, context)
return jsonify(result)

-----------------------------

HEALTH CHECK

-----------------------------

@app.route('/') def home(): return "Campbell Cognitive Pipeline API running"

if name == 'main': app.run(debug=True, host='0.0.0.0', port=5000)
