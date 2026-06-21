import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document


# -----------------------
# Read JD
# -----------------------

doc = Document("data/job_description.docx")

jd_text = ""

for para in doc.paragraphs:
    jd_text += para.text + "\n"


# -----------------------
# Build Candidate Text
# -----------------------

def build_candidate_text(candidate):

    profile = candidate["profile"]

    text = []

    text.append(profile.get("headline", ""))
    text.append(profile.get("summary", ""))

    skills = [
        skill["name"]
        for skill in candidate["skills"]
    ]

    text.append("Skills: " + ", ".join(skills))

    for job in candidate["career_history"]:
        text.append(job.get("title", ""))
        text.append(job.get("description", ""))

    return "\n".join(text)


# -----------------------
# Load First Candidate
# -----------------------

with open(
    "data/candidates.jsonl",
    "r",
    encoding="utf-8"
) as f:

    candidate = json.loads(next(f))

candidate_text = build_candidate_text(candidate)


# -----------------------
# Embeddings
# -----------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

jd_embedding = model.encode([jd_text])

candidate_embedding = model.encode(
    [candidate_text]
)

score = cosine_similarity(
    jd_embedding,
    candidate_embedding
)

print("Similarity Score:")
print(score[0][0])