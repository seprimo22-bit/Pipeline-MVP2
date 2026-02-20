from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Simple domain gate

def classify_question(q):
    private_keywords = [
        "paper zero",
        "titan alloy",
        "third harmonic",
        "campbell",
        "your book",
        "your theory"
    ]

    q = q.lower()
    for k in private_keywords:
        if k in q:
            return "private"

    return "general"


def general_answer(q):
    return {
        "source": "general knowledge",
        "answer":
        "This response is generated from general knowledge. "
        "Private documents were not used."
    }


def private_answer(q):
    return {
        "source": "private documents",
        "answer":
        "This question relates to your internal documents. "
        "Retrieval would occur here without exposing documents."
    }


@app.route("/", methods=["GET","POST"])
def home():
    if request.method == "POST":
        q = request.form["question"]

        domain = classify_question(q)

        if domain == "private":
            result = private_answer(q)
        else:
            result = general_answer(q)

        return jsonify(result)

    return render_template_string("""
    <h2>Campbell Cognitive Pipeline</h2>
    <form method="post">
    <input name="question" style="width:400px;">
    <button>Ask</button>
    </form>
    """)


if __name__ == "__main__":
    app.run(debug=True)
