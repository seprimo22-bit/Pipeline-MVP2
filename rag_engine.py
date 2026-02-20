import os
from openai import OpenAI

# Initialize OpenAI client safely
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_article_facts(article_text, question=None):
    """
    Extract facts ONLY from article text.
    Question is used ONLY for relevance guidance.
    """

    if not article_text or len(article_text.strip()) < 50:
        return {
            "established_facts": [],
            "article_facts": [],
            "unknowns": ["No article text provided or text too short."]
        }

    prompt = f"""
You are a FACT EXTRACTION ENGINE.

STRICT RULES:
1. ONLY extract facts explicitly stated in the article text.
2. DO NOT infer, speculate, or answer the question.
3. The question is ONLY context.
4. Separate facts into:

- Established Scientific Facts
- Article-Specific Facts
- Unknowns / Limits

USER QUESTION (context only):
{question if question else "None"}

ARTICLE TEXT:
{article_text}

Return bullet points only.
No commentary.
No extra explanation.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "Fact extraction only."},
                {"role": "user", "content": prompt},
            ],
        )

        return {
            "raw_output": response.choices[0].message.content
        }

    except Exception as e:
        return {
            "error": str(e)
        }


def run_fact_pipeline(article_text, question=None):
    """
    Main pipeline entry point for Flask app.
    Keeps future compatibility with multi-stage pipeline.
    """

    facts = extract_article_facts(article_text, question)

    return {
        "status": "success",
        "pipeline": "fact_only_article_analysis",
        "result": facts,
    }
