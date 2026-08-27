from pdf_loader import load_pdfs

def chunk_document(document):

    chunk_size = 300
    chunk_overlap = 50

    chunks = []
    start = 0
    while start < len(document):
        end = start + chunk_size
        chunk = document[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

document = load_pdfs("data/RealEstate")
chunks = chunk_document(document)
print("Total chunks:", len(chunks))

print("\nFirst Chunk:", chunks[1])