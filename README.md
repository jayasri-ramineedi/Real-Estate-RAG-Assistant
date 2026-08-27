# 🏠 Real Estate RAG Assistant

An AI-powered Real Estate Question Answering system built using
Retrieval-Augmented Generation (RAG).

# 📌 Project Overview

This application allows users to ask questions about real estate
documents and receive accurate answers based on the information
available in those documents.

The system uses document chunking, embeddings, FAISS vector search,
and Google Gemini for answer generation.

# 🚀 Features

- 📄 PDF document processing
- ✂️ Text chunking
- 🔢 Text embeddings
- 🔎 Semantic search using FAISS
- 🤖 Google Gemini for answer generation
- 📊 RAG evaluation metrics
- 🌐 Streamlit web interface

# 🛠️ Technologies Used

- Python
- Streamlit
- Google Gemini API
- Sentence Transformers
- FAISS
- NumPy
- PyPDF

# 🔄 RAG Workflow

PDF Documents  
↓  
Text Extraction  
↓  
Chunking  
↓  
Embeddings  
↓  
FAISS Vector Database  
↓  
Question Embedding  
↓  
Similarity Search  
↓  
Relevant Context  
↓  
Google Gemini  
↓  
Final Answer

# 📂 Project Structure

```text
RAG Project/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── RealEstate/
│       ├── 01_SkyMeadows_Price_List.pdf
│       ├── 02_EmeraldEnclave_Handbook.pdf
│       ├── 03_Listing_Portfolio.pdf
│       └── 04_Maintenance_Compliance_Log.pdf
│
└── src/
    ├── config.py
    ├── pdf_loader.py
    ├── chunker.py
    ├── embeddings.py
    ├── vector_store.py
    ├── generator.py
    ├── evaluation.py
    └── query.py