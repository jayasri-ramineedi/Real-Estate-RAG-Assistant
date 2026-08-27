from config import client


def generate_answer(question, context):

    prompt = f"""
You are a helpful real estate assistant.

Answer the question ONLY using the information provided
in the context below.

If the answer is not available in the context, say:
"I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

Give a concise and accurate answer.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text