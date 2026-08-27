from google import genai
from config import client


def evaluate_answer(question, context, answer):

    prompt = f"""
You are evaluating a RAG system.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate the answer using these metrics:

1. Faithfulness: Is the answer supported by the retrieved context?
2. Relevance: Does the answer directly answer the question?
3. Context Relevance: Is the retrieved context useful for answering the question?

Give a score from 0 to 10 for each metric.

Return ONLY in this format:

Faithfulness: X/10
Relevance: X/10
Context Relevance: X/10
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text