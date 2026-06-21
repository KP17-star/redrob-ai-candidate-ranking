import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# JOB DESCRIPTION
# =========================

jd_text = """
AI Engineer
LLM
Ranking Systems
Retrieval
Embeddings
Fine-tuning
Machine Learning
Python
Search
Recommendation Systems
Vector Databases
RAG
"""

jd_embedding = model.encode(jd_text)

scores = []

print("Processing candidates...")

# =========================
# READ CANDIDATES
# =========================

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:

    count = 0

    for line in f:

        candidate = json.loads(line)

        profile = candidate.get("profile", {})

        headline = profile.get("headline", "")
        summary = profile.get("summary", "")

        # =========================
        # SKILLS
        # =========================

        skills_text = ""

        for skill in candidate.get("skills", []):
            skills_text += skill.get("name", "") + " "

        # =========================
        # CAREER HISTORY
        # =========================

        career_text = ""

        for job in candidate.get("career_history", []):

            career_text += job.get("title", "") + " "

            career_text += job.get("description", "") + " "

        # =========================
        # FULL CANDIDATE TEXT
        # =========================

        candidate_text = f"""
        {headline}

        {summary}

        Skills:
        {skills_text}

        Experience:
        {career_text}
        """

        candidate_embedding = model.encode(candidate_text)

        score = cosine_similarity(
            [jd_embedding],
            [candidate_embedding]
        )[0][0]

        scores.append(
            (
                candidate["candidate_id"],
                float(score),
                headline
            )
        )

        count += 1

        # First 1000 candidates only
        if count == 1000:
            break

# =========================
# SORT
# =========================

scores.sort(
    key=lambda x: x[1],
    reverse=True
)

# =========================
# DISPLAY RESULTS
# =========================

print("\nTOP 20 CANDIDATES\n")

for candidate_id, score, headline in scores[:20]:

    print(
        f"{candidate_id} | "
        f"{score:.4f} | "
        f"{headline}"
    )