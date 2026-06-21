import json

def build_candidate_text(candidate):
    profile = candidate["profile"]

    text = []

    text.append(profile.get("headline", ""))
    text.append(profile.get("summary", ""))

    text.append(
        f"Experience: {profile.get('years_of_experience', 0)} years"
    )

    skills = [
        skill["name"]
        for skill in candidate["skills"]
    ]

    text.append("Skills: " + ", ".join(skills))

    for job in candidate["career_history"]:
        text.append(job.get("title", ""))
        text.append(job.get("description", ""))

    return "\n".join(text)


with open(
    "data/candidates.jsonl",
    "r",
    encoding="utf-8"
) as f:

    first_candidate = json.loads(next(f))

candidate_text = build_candidate_text(first_candidate)

print(candidate_text[:3000])