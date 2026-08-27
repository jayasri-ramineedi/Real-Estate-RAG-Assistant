import faiss
import numpy as np

from embeddings import create_embeddings
from chunker import chunk_document
from pdf_loader import load_pdfs


# Load documents
document = load_pdfs("data/RealEstate")


# Create chunks
chunks = chunk_document(document)

print("Total chunks:", len(chunks))


# Create embeddings
embeddings = create_embeddings(chunks)

print("Embedding shape:", embeddings.shape)


def create_faiss_index(embeddings):

    # Convert embeddings to float32
    embeddings = np.array(embeddings).astype("float32")

    # Get embedding dimension
    dimension = embeddings.shape[1]

    print("Embedding dimension:", dimension)

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to FAISS
    index.add(embeddings)

    print("FAISS index created.")
    print("Total vectors stored:", index.ntotal)

    return index


def search_faiss(question_embedding, index, k=3):

    # Search FAISS
    distances, indices = index.search(
        question_embedding,
        k
    )

    return distances[0], indices[0]


# Create FAISS index
index = create_faiss_index(embeddings)

# Save FAISS index

faiss.write_index(index, "realestate.index")

print("FAISS index saved successfully.")