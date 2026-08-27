import os
from pypdf import PdfReader

def load_pdfs(data_dir):
    document = ""
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(data_dir, file_name)
            reader = PdfReader(file_path)

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    document += text +"\n"

            print("Document loaded:", file_name)
    return document
document = load_pdfs("data/RealEstate")
print("Total characters:", len(document))