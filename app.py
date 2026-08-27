import sys
sys.path.append("src")

import streamlit as st
import faiss

from src.embeddings import create_embeddings
from src.vector_store import search_faiss
from src.chunker import chunk_document
from src.pdf_loader import load_pdfs
from src.generator import generate_answer
from src.evaluation import evaluate_answer


# Page configuration
st.set_page_config(
    page_title="Real Estate RAG Assistant",
    page_icon="🏠",
    layout="centered"
)


# Styling
st.markdown("""
<style>
h1 {
    text-align: center;
}

.stButton > button {
    width: 100%;
    border-radius: 8px;
    font-weight: bold;
}

.answer-box {
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)


# Title
st.title("🏠 Real Estate RAG Assistant")
st.write("Ask questions about your real estate documents.")

st.divider()


# Load documents only once
@st.cache_data
def load_chunks():

    document = load_pdfs("data/RealEstate")
    chunks = chunk_document(document)

    return chunks


# Load FAISS index only once
@st.cache_resource
def load_faiss_index():

    index = faiss.read_index("realestate.index")

    return index


chunks = load_chunks()
index = load_faiss_index()


# Question
question = st.text_input("🔎 Enter your question:")


if st.button("Ask"):

    if question:

        # Embed only the question
        question_embedding = create_embeddings([question])

        # Search FAISS
        distances, indices = search_faiss(
            question_embedding,
            index,
            k=3
        )

        # Retrieve chunks
        retrieved_chunks = []

        for i in indices:
            retrieved_chunks.append(chunks[i])

        # Create context
        context = "\n\n".join(retrieved_chunks)

        # Generate answer
        answer = generate_answer(question, context)

        st.subheader("💡 Answer")

        st.markdown(
            f'<div class="answer-box">{answer}</div>',
            unsafe_allow_html=True
        )

        # Evaluation
        evaluation = evaluate_answer(
            question,
            context,
            answer
        )

        st.subheader("📊 Evaluation Metrics")
        st.write(evaluation)

    else:

        st.warning("Please enter a question.")