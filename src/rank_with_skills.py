import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

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

important_skills = {
    "Python",
    "NLP",
    "Fine-tuning LLMs",
    "Milvus",
    "AWS",
    "Flask",
    "BentoML",
    "GCP",
    "Speech Recognition",
    "Image Classification"
}

jd_embedding = model.encode(jd_text)

scores = []

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:

    count = 0

    for line in f:

        candidate = json.loads(line)

        profile = candidate.get("profile", {})

        headline = profile.get("headline", "")
        summary = profile.get("summary", "")

        skills_text = ""
        skill_bonus = 0

        for skill in candidate.get("skills", []):

            skill_name = skill.get("name", "")

            skills_text += skill_name + " "

            if skill_name in important_skills:
                skill_bonus += 0.02

        career_text = ""

        for job in candidate.get("career_history", []):

            career_text += job.get("title", "") + " "
            career_text += job.get("description", "") + " "

        candidate_text = f"""
        {headline}

        {summary}

        Skills:
        {skills_text}

        Experience:
        {career_text}
        """

        candidate_embedding = model.encode(candidate_text)

        semantic_score = cosine_similarity(
            [jd_embedding],
            [candidate_embedding]
        )[0][0]

        final_score = semantic_score + skill_bonus

        scores.append(
            (
                candidate["candidate_id"],
                final_score,
                headline
            )
        )

        count += 1

        if count == 1000:
            break

scores.sort(
    key=lambda x: x[1],
    reverse=True
)

print("\nTOP 20 CANDIDATES\n")

for candidate_id, score, headline in scores[:20]:

    print(
        f"{candidate_id} | "
        f"{score:.4f} | "
        f"{headline}"
    )