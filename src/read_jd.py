from docx import Document

doc = Document("data/job_description.docx")

text = ""

for para in doc.paragraphs:
    text += para.text + "\n"

print(text[:5000])