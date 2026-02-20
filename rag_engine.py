import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_article_facts(article_text, question=None):
    """
    Extract facts ONLY from article text.
    Question is used ONLY to guide relevance,
    not as a source of facts.
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
2. DO NOT invent, infer, or hypothesize.
3. DO NOT answer the user's question directly.
4. The question is ONLY for context relevance.
5. Separate facts into 3 categories:

- Established Scientific Facts:
  Widely accepted scientific facts referenced.

- Article-Specific Facts:
  Claims or findings stated directly in this article.

- Unknowns / Limits:
  What the article does NOT prove,
  assumptions, missing info, or uncertainty.

USER QUESTION (context only):
{question if question else "None"}

ARTICLE TEXT:
{article_text}

Return clean bullet points.
No commentary.
No explanation.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "Fact extraction only."},
            {"role": "user", "content": prompt},
        ],
    )

    text = response.choices[0].message.content

    return {"raw_output": text}
