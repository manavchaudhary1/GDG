from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class CareerInsights:
    market_alignment_score: int
    target_role: str
    strengths: list[str]
    skill_gaps: list[str]
    recommendations: list[str]


def _normalize_skills(skills: list[str] | None) -> list[str]:
    if not skills:
        return []
    return [skill.strip().lower() for skill in skills if isinstance(skill, str) and skill.strip()]


def _job_match_score(candidate_skills: set[str], job_skills: list[str]) -> float:
    normalized_job_skills = _normalize_skills(job_skills)
    if not normalized_job_skills:
        return 0.0

    overlap = candidate_skills.intersection(normalized_job_skills)
    return len(overlap) / len(set(normalized_job_skills))


def generate_career_insights(
    profile: dict[str, Any],
    jobs: list[dict[str, Any]],
    trends: dict[str, Any] | None = None,
) -> CareerInsights:
    """Produce personalized career guidance from profile, job postings, and market trends."""
    trends = trends or {}

    target_role = str(profile.get("target_role", "")).strip() or "unspecified"
    candidate_skills = set(_normalize_skills(profile.get("skills")))

    scoped_jobs = [
        job for job in jobs if target_role == "unspecified" or target_role.lower() in str(job.get("title", "")).lower()
    ]
    if not scoped_jobs:
        scoped_jobs = jobs

    demand_counter: Counter[str] = Counter()
    job_scores: list[float] = []

    for job in scoped_jobs:
        job_skills = _normalize_skills(job.get("skills"))
        demand_counter.update(job_skills)
        job_scores.append(_job_match_score(candidate_skills, job_skills))

    emerging_skills = trends.get("emerging_skills", {})
    if isinstance(emerging_skills, dict):
        for skill, weight in emerging_skills.items():
            if isinstance(skill, str) and skill.strip():
                demand_counter[skill.strip().lower()] += int(weight) if isinstance(weight, (int, float)) else 1

    strengths = sorted(candidate_skills.intersection(demand_counter.keys()))
    missing_ranked = [skill for skill, _ in demand_counter.most_common() if skill not in candidate_skills]

    alignment_score = int(round(mean(job_scores) * 100)) if job_scores else 0

    recommendations: list[str] = []
    if missing_ranked:
        upskill_targets = ", ".join(missing_ranked[:3])
        recommendations.append(f"Prioritize learning: {upskill_targets}.")
    if scoped_jobs and alignment_score < 60:
        recommendations.append("Improve role alignment by tailoring projects and resume bullets to target job requirements.")
    if not recommendations:
        recommendations.append("Maintain momentum by deepening expertise and applying consistently to well-matched roles.")

    return CareerInsights(
        market_alignment_score=alignment_score,
        target_role=target_role,
        strengths=strengths,
        skill_gaps=missing_ranked[:5],
        recommendations=recommendations,
    )
