from embeddings import create_embeddings

#Embed the Question
#question = "What is the price of a 3 BHK in Sky Meadows?"
question=input('Enter a question:')
question_embedding = create_embeddings([question])
print("Question embedding shape:", question_embedding.shape)

from vector_store import create_faiss_index, search_faiss
from chunker import chunk_document
from pdf_loader import load_pdfs

#Load documents
document = load_pdfs("data/RealEstate")

#Create chunks
chunks = chunk_document(document)

#Create embeddings
embeddings = create_embeddings(chunks)

#Create FAISS index
index = create_faiss_index(embeddings)

distances, indices = search_faiss(question_embedding,
                                  index,
                                  k=3)
print("Distances:", distances)
print("Chunk numbers:", indices)

#Retrieve the Actual Chunks
def retrieve_chunks(indices, chunks):
    retrieved_chunks = []

    for i in indices:
        retrieved_chunks.append(chunks[i])
    return retrieved_chunks

retrieved_chunks = retrieve_chunks(indices, chunks)

print("Number of retrieved chunks:", len(retrieved_chunks))

for i, chunk in enumerate(retrieved_chunks):
    print("\n--- Chunk", i + 1, "---")
    print(chunk)

#Create Context
def create_context(retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    return context

context = create_context(retrieved_chunks)

print("\nCONTEXT:")
print(context)

# Step 9 — Generate Answer

from generator import generate_answer

answer = generate_answer(question, context)

print("\nANSWER:")
print(answer)

#Load saved FAISS index
import faiss

index = faiss.read_index("realestate.index")

print("FAISS index loaded.")
print("Total vectors:", index.ntotal)