import json
import pickle
import numpy as np

from resume_parser.extract_resume import extract_resume_text
from agents.resume_analyzer import analyze_resume
from agents.skill_matcher import analyze_skill_gap
from scoring.ats_score import predict_ats
from job_matcher.linkedin_jobs import match_jobs


def predict_job_role(analysis):

    model = pickle.load(open("models/job_role_model.pkl", "rb"))
    encoder = pickle.load(open("models/job_role_encoder.pkl", "rb"))

    skills = []

    if isinstance(analysis["skills"], dict):
        for v in analysis["skills"].values():
            skills.extend(v)
    else:
        skills = analysis["skills"]

    features = np.array([[
        len(skills),
        len(analysis["experience_years"]),
        2,
        len(analysis["certification"]),
        len(analysis["projects"])
    ]])

    role_id = model.predict(features)[0]

    role = encoder.inverse_transform([role_id])[0]

    return role


def main():

    file_path = "data/resume3.pdf"

    print("\nExtracting resume text...\n")

    resume_text = extract_resume_text(file_path)

    print("Analyzing resume using CrewAI...\n")

    analysis = analyze_resume(resume_text)

    with open("outputs/resume_analysis_output.json", "w") as f:
        json.dump(analysis, f, indent=4)

    # ATS score
    ats_score = predict_ats(analysis)

    print("\nATS Score:", ats_score)

    # Predict best job role
    best_role = predict_job_role(analysis)

    print("\nBest Job Role for this Resume:", best_role)

    location = input("\nEnter job location: ")

    # Skill gap analysis
    print("\nAnalyzing skill gap...\n")

    skill_analysis = analyze_skill_gap(analysis["skills"], best_role)

    with open("outputs/skill_gap_output.json", "w") as f:
        json.dump(skill_analysis, f, indent=4)

    # Show recommended skills
    print("\nRecommended Skills To Add:\n")

    for skill in skill_analysis["recommended_skills_to_add"]:
        print("-", skill)

    # Show YouTube learning links
    print("\nLearning Resources:\n")

    for resource in skill_analysis["learning_resources"]:
        print(resource["skill"], "→", resource["youtube_video"])

    # Job recommendations
    print("\nFinding matching jobs...\n")

    jobs = match_jobs(analysis["skills"], best_role, location)

    print("\nTop Jobs:\n")

    for job in jobs:
        print(job)


if __name__ == "__main__":
    main()