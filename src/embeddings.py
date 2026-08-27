import numpy as np
from sentence_transformers import SentenceTransformer
from pdf_loader import load_pdfs
from chunker import chunk_document

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Local embedding model loaded.")

def create_embeddings(chunks):

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )
    return np.array(
        embeddings,
        dtype = "float32"
    )
    

document = load_pdfs("data/RealEstate")

chunks = chunk_document(document)

chunk_embeddings = create_embeddings(chunks)

print("Embedding shape:", chunk_embeddings.shape)

print("\nFirst embedding:", chunk_embeddings[0])