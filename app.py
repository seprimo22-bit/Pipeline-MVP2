from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ---------------------------------------------------
# MAIN PAGE (Simple built-in interface)
# ---------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    if request.method == "POST":
        question = request.form.get("question")
        article = request.form.get("article")

        result = run_pipeline(question, article)

    return render_template_string("""
    <html>
    <head>
        <title>Campbell Cognitive Pipeline</title>
        <style>
            body { font-family: Arial; margin: 40px; }
            textarea { width: 100%; height: 120px; }
            input[type=submit] { padding: 10px 20px; }
            .result { margin-top: 20px; padding: 10px; background:#eee; }
        </style>
    </head>
    <body>

        <h1>Campbell Cognitive Pipeline</h1>

        <form method="POST">

            <h3>Question (Optional)</h3>
            <textarea name="question"
                placeholder="Ask a question..."></textarea>

            <h3>Article Text (Optional)</h3>
            <textarea name="article"
                placeholder="Paste article text here..."></textarea>

            <br><br>
            <input type="submit" value="Run Pipeline">

        </form>

        {% if result %}
        <div class="result">
            <h3>Pipeline Output</h3>
            <pre>{{ result }}</pre>
        </div>
        {% endif %}

    </body>
    </html>
    """, result=result)


# ---------------------------------------------------
# PIPELINE CORE LOGIC
# ---------------------------------------------------
def run_pipeline(question=None, article_text=None):

    question = (question or "").strip()
    article_text = (article_text or "").strip()

    # Question only
    if question and not article_text:
        return knowledge_mode(question)

    # Article only
    elif article_text and not question:
        return article_mode(article_text)

    # Both together
    elif question and article_text:
        return combined_mode(question, article_text)

    # Nothing entered
    else:
        return "Enter a question, an article, or both."


# ---------------------------------------------------
# MODES
# ---------------------------------------------------
def knowledge_mode(question):
    return f"""
MODE: Question Only

Question:
{question}

Action:
General reasoning / baseline knowledge mode activated.
"""


def article_mode(article):
    return f"""
MODE: Article Only

Article Length:
{len(article)} characters

Action:
Fact extraction / article analysis mode.
"""


def combined_mode(question, article):
    return f"""
MODE: Combined Analysis

Question:
{question}

Article Length:
{len(article)} characters

Action:
Comparative reasoning between question and article.
"""


# ---------------------------------------------------
# RUN SERVER
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
