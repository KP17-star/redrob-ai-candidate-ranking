import json
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 100


def build_candidate_text(candidate):
    profile = candidate.get("profile", {})

    parts = []

    headline = profile.get("headline", "")
    if headline:
        parts.append(headline)

    summary = profile.get("summary", "")
    if summary:
        parts.append(summary)

    skills = profile.get("skills", [])
    if skills:
        skill_names = []

        for skill in skills:
            if isinstance(skill, dict):
                name = skill.get("skill_name") or skill.get("name")
                if name:
                    skill_names.append(name)
            elif isinstance(skill, str):
                skill_names.append(skill)

        if skill_names:
            parts.append("Skills: " + ", ".join(skill_names))

    experiences = candidate.get("experience", [])

    for exp in experiences[:3]:
        title = exp.get("title", "")
        description = exp.get("description", "")

        text = f"{title} {description}".strip()

        if text:
            parts.append(text)

    return "\n".join(parts)


def build_reasoning(candidate, score):
    profile = candidate.get("profile", {})

    headline = profile.get("headline", "No headline")

    skills = profile.get("skills", [])
    skill_count = len(skills)

    experience = candidate.get("total_experience_years", "N/A")

    return (
        f"{headline}; "
        f"{experience} yrs experience; "
        f"{skill_count} skills; "
        f"semantic match score {score:.3f}"
    )


def main():

    jd_text = """
    Senior AI Engineer

    Skills:
    machine learning, LLMs, retrieval, ranking, embeddings,
    recommendation systems, vector databases, fine-tuning,
    Python, NLP, search systems, RAG, evaluation frameworks
    """

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading candidates...")

    candidates = []

    with open(
        "data/candidates.jsonl",
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            candidates.append(json.loads(line))

    print(f"Loaded {len(candidates)} candidates")

    print("Building candidate text...")

    candidate_texts = [
        build_candidate_text(candidate)
        for candidate in candidates
    ]

    print("Creating JD embedding...")

    jd_embedding = model.encode(
        jd_text,
        normalize_embeddings=True
    )

    print("Creating candidate embeddings...")

    candidate_embeddings = model.encode(
        candidate_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print("Calculating similarity scores...")

    scores = cosine_similarity(
        [jd_embedding],
        candidate_embeddings
    )[0]

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    results = []

    for rank, idx in enumerate(top_indices, start=1):

        candidate = candidates[idx]
        score = float(scores[idx])

        results.append({
            "candidate_id": candidate["candidate_id"],
            "rank": rank,
            "score": round(score, 4),
            "reasoning": build_reasoning(candidate, score)
        })

    submission = pd.DataFrame(results)

    os.makedirs("outputs", exist_ok=True)

    output_path = "outputs/submission.csv"

    submission.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print("\nTop 5 candidates:\n")
    print(submission.head())


if __name__ == "__main__":
    main()