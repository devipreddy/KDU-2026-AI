"""
Utilities for explainable candidate ranking and evidence generation.
"""

import os
import re
import uuid
from typing import Dict, List

from utils.langchain_utils import (
    assess_resume_for_job,
    extract_match_score,
    find_relevant_sections,
    prepare_resume_documents,
)
from utils.resume_utils import read_resume

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "will",
    "you",
    "your",
}

SCORING_PROFILES = {
    "balanced": {
        "semantic_score": 0.35,
        "skill_coverage": 0.25,
        "experience_alignment": 0.15,
        "education_alignment": 0.10,
        "llm_assessment_score": 0.15,
    },
    "skills_first": {
        "semantic_score": 0.25,
        "skill_coverage": 0.35,
        "experience_alignment": 0.10,
        "education_alignment": 0.10,
        "llm_assessment_score": 0.20,
    },
    "experience_first": {
        "semantic_score": 0.25,
        "skill_coverage": 0.20,
        "experience_alignment": 0.30,
        "education_alignment": 0.10,
        "llm_assessment_score": 0.15,
    },
    "entry_level": {
        "semantic_score": 0.30,
        "skill_coverage": 0.25,
        "experience_alignment": 0.10,
        "education_alignment": 0.20,
        "llm_assessment_score": 0.15,
    },
}


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _tokenize(text):
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+#.\-]{1,}", (text or "").lower())
    return {token for token in tokens if token not in STOPWORDS}


def _split_into_sections(text, chunk_size=900):
    paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", text or "") if segment.strip()]
    if paragraphs:
        return paragraphs

    compact = _normalize_whitespace(text)
    return [compact[i : i + chunk_size] for i in range(0, len(compact), chunk_size) if compact[i : i + chunk_size]]


def _lexical_evidence(resume_text, job_description, top_k=3):
    job_tokens = _tokenize(job_description)
    sections = _split_into_sections(resume_text)
    scored_sections = []

    for section in sections:
        section_tokens = _tokenize(section)
        if not section_tokens:
            continue

        overlap = len(job_tokens & section_tokens)
        denominator = max(1, len(job_tokens))
        confidence = overlap / denominator
        scored_sections.append((_normalize_whitespace(section), confidence))

    scored_sections.sort(key=lambda item: item[1], reverse=True)
    return scored_sections[:top_k]


def _compute_skill_coverage(job_description, resume_text):
    job_tokens = _tokenize(job_description)
    resume_tokens = _tokenize(resume_text)
    if not job_tokens:
        return 0.0, [], []

    overlapping = sorted(job_tokens & resume_tokens)
    missing = sorted(job_tokens - resume_tokens)
    score = len(overlapping) / len(job_tokens)
    return score, overlapping[:20], missing[:20]


def _extract_year_requirements(text):
    matches = re.findall(r"(\d+)\+?\s*(?:years|yrs)", text or "", flags=re.IGNORECASE)
    if not matches:
        return None
    return max(int(match) for match in matches)


def _compute_experience_alignment(job_description, resume_text):
    required_years = _extract_year_requirements(job_description)
    candidate_years = _extract_year_requirements(resume_text)

    if required_years is None:
        return 0.7
    if candidate_years is None:
        return 0.4

    return min(1.0, candidate_years / max(1, required_years))


def _compute_education_alignment(job_description, resume_text):
    degree_keywords = ("bachelor", "master", "phd", "degree", "b.tech", "m.tech")
    job_requires_degree = any(keyword in (job_description or "").lower() for keyword in degree_keywords)
    resume_has_degree = any(keyword in (resume_text or "").lower() for keyword in degree_keywords)

    if not job_requires_degree:
        return 0.7

    return 1.0 if resume_has_degree else 0.3


def _weighted_average(score_map, scoring_profile):
    weights = SCORING_PROFILES.get(scoring_profile, SCORING_PROFILES["balanced"])
    weighted_sum = 0.0
    total_weight = 0.0

    for key, weight in weights.items():
        value = score_map.get(key)
        if value is None:
            continue

        weighted_sum += value * weight
        total_weight += weight

    if not total_weight:
        return 0.0

    return weighted_sum / total_weight


def _confidence_band(average_evidence_score, llm_used):
    if average_evidence_score >= 0.65 and llm_used:
        return "high"
    if average_evidence_score >= 0.4:
        return "medium"
    return "low"


def _provider_mode(embeddings, llm):
    if embeddings and llm:
        return "full_fidelity"
    if embeddings:
        return "retrieval_only"
    if llm:
        return "llm_without_embeddings"
    return "heuristic_only"


def _retrieve_relevant_sections(resume_text, filename, job_description, embeddings, top_k=3):
    if embeddings:
        processed_resume = prepare_resume_documents(resume_text, filename)
        return find_relevant_sections(processed_resume, job_description, embeddings)[:top_k]

    return _lexical_evidence(resume_text, job_description, top_k=top_k)


def build_candidate_analysis(
    file_path,
    resume_dir,
    job_description,
    embeddings,
    llm,
    scoring_profile="balanced",
):
    resume_text = read_resume(file_path, resume_dir)
    if not resume_text:
        raise ValueError(f"Failed to read resume: {file_path}")

    filename = os.path.basename(file_path)
    relevant_sections = _retrieve_relevant_sections(
        resume_text,
        filename,
        job_description,
        embeddings,
        top_k=3,
    )

    semantic_score = sum(score for _, score in relevant_sections) / max(1, len(relevant_sections))
    skill_coverage, matched_keywords, missing_keywords = _compute_skill_coverage(job_description, resume_text)
    experience_alignment = _compute_experience_alignment(job_description, resume_text)
    education_alignment = _compute_education_alignment(job_description, resume_text)

    llm_assessment = None
    llm_score = None
    if llm:
        llm_assessment = assess_resume_for_job(resume_text, job_description, llm)
        parsed_score = extract_match_score(llm_assessment)
        if parsed_score is not None:
            llm_score = parsed_score / 100

    score_map = {
        "semantic_score": semantic_score,
        "skill_coverage": skill_coverage,
        "experience_alignment": experience_alignment,
        "education_alignment": education_alignment,
        "llm_assessment_score": llm_score,
    }
    final_score = _weighted_average(score_map, scoring_profile)

    evidence = [
        {
            "evidence_id": f"{os.path.splitext(filename)[0]}-evidence-{index}",
            "candidate_file": file_path,
            "source": filename,
            "section_index": index,
            "section": _normalize_whitespace(section)[:700],
            "score": round(score, 4),
        }
        for index, (section, score) in enumerate(relevant_sections, start=1)
    ]

    return {
        "candidate_file": file_path,
        "candidate_name": os.path.splitext(filename)[0].replace("-", " ").replace("_", " ").title(),
        "overall_score": round(final_score * 100, 2),
        "component_scores": {
            "semantic_score": round(semantic_score * 100, 2),
            "skill_coverage": round(skill_coverage * 100, 2),
            "experience_alignment": round(experience_alignment * 100, 2),
            "education_alignment": round(education_alignment * 100, 2),
            "llm_assessment_score": round(llm_score * 100, 2) if llm_score is not None else None,
        },
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "confidence": _confidence_band(semantic_score, llm is not None),
        "mode": _provider_mode(embeddings, llm),
        "evidence": evidence,
        "assessment": llm_assessment,
    }


def rank_candidates(
    job_description,
    candidate_files,
    resume_dir,
    embeddings,
    llm,
    top_k=5,
    scoring_profile="balanced",
):
    if not candidate_files:
        raise ValueError("At least one candidate file is required for ranking.")

    analyses = [
        build_candidate_analysis(
            file_path=file_path,
            resume_dir=resume_dir,
            job_description=job_description,
            embeddings=embeddings,
            llm=llm,
            scoring_profile=scoring_profile,
        )
        for file_path in candidate_files
    ]

    analyses.sort(key=lambda item: item["overall_score"], reverse=True)
    ranked_candidates = []
    for index, analysis in enumerate(analyses[:top_k], start=1):
        analysis["rank"] = index
        ranked_candidates.append(analysis)

    return {
        "run_id": str(uuid.uuid4()),
        "job_description": job_description,
        "candidate_count": len(candidate_files),
        "top_k": min(top_k, len(candidate_files)),
        "scoring_profile": scoring_profile,
        "ranked_candidates": ranked_candidates,
    }


def compare_candidates(job_description, candidate_files, resume_dir, embeddings, llm, scoring_profile="balanced"):
    ranking = rank_candidates(
        job_description=job_description,
        candidate_files=candidate_files,
        resume_dir=resume_dir,
        embeddings=embeddings,
        llm=llm,
        top_k=len(candidate_files),
        scoring_profile=scoring_profile,
    )

    candidates = ranking["ranked_candidates"]
    if len(candidates) < 2:
        raise ValueError("At least two candidates are required for comparison.")

    winner = candidates[0]
    comparison_matrix = []
    for candidate in candidates:
        comparison_matrix.append(
            {
                "candidate_file": candidate["candidate_file"],
                "overall_score": candidate["overall_score"],
                "semantic_score": candidate["component_scores"]["semantic_score"],
                "skill_coverage": candidate["component_scores"]["skill_coverage"],
                "experience_alignment": candidate["component_scores"]["experience_alignment"],
                "education_alignment": candidate["component_scores"]["education_alignment"],
            }
        )

    return {
        "run_id": ranking["run_id"],
        "winner": winner["candidate_file"],
        "winner_score": winner["overall_score"],
        "comparison_matrix": comparison_matrix,
        "key_differentiators": [
            {
                "candidate_file": candidate["candidate_file"],
                "matched_keywords": candidate["matched_keywords"][:10],
                "missing_keywords": candidate["missing_keywords"][:10],
                "top_evidence": candidate["evidence"][:2],
            }
            for candidate in candidates
        ],
    }


def explain_candidate_fit(file_path, job_description, resume_dir, embeddings, llm, scoring_profile="balanced"):
    analysis = build_candidate_analysis(
        file_path=file_path,
        resume_dir=resume_dir,
        job_description=job_description,
        embeddings=embeddings,
        llm=llm,
        scoring_profile=scoring_profile,
    )

    return {
        "candidate_file": analysis["candidate_file"],
        "candidate_name": analysis["candidate_name"],
        "overall_score": analysis["overall_score"],
        "confidence": analysis["confidence"],
        "matched_keywords": analysis["matched_keywords"],
        "missing_keywords": analysis["missing_keywords"],
        "evidence": analysis["evidence"],
        "assessment": analysis["assessment"],
        "rationale": (
            f"{analysis['candidate_name']} scored {analysis['overall_score']} with "
            f"strongest signals in semantic alignment and keyword coverage."
        ),
    }


def show_candidate_evidence(file_path, job_description, resume_dir, embeddings):
    resume_text = read_resume(file_path, resume_dir)
    if not resume_text:
        raise ValueError(f"Failed to read resume: {file_path}")

    filename = os.path.basename(file_path)
    evidence = _retrieve_relevant_sections(resume_text, filename, job_description, embeddings, top_k=5)
    return {
        "candidate_file": file_path,
        "job_description": job_description,
        "evidence": [
            {
                "candidate_file": file_path,
                "source": os.path.basename(file_path),
                "section_index": index,
                "section": _normalize_whitespace(section)[:700],
                "score": round(score * 100, 2),
            }
            for index, (section, score) in enumerate(evidence, start=1)
        ],
    }


def generate_shortlist(job_description, candidate_files, resume_dir, embeddings, llm, top_k=3, scoring_profile="balanced"):
    ranking = rank_candidates(
        job_description=job_description,
        candidate_files=candidate_files,
        resume_dir=resume_dir,
        embeddings=embeddings,
        llm=llm,
        top_k=top_k,
        scoring_profile=scoring_profile,
    )

    shortlist = []
    for candidate in ranking["ranked_candidates"]:
        shortlist.append(
            {
                "rank": candidate["rank"],
                "candidate_file": candidate["candidate_file"],
                "candidate_name": candidate["candidate_name"],
                "overall_score": candidate["overall_score"],
                "confidence": candidate["confidence"],
                "matched_keywords": candidate["matched_keywords"][:10],
                "missing_keywords": candidate["missing_keywords"][:10],
                "shortlist_rationale": (
                    f"Selected at rank {candidate['rank']} with score {candidate['overall_score']} "
                    f"and evidence-backed alignment on {', '.join(candidate['matched_keywords'][:5]) or 'core requirements'}."
                ),
            }
        )

    recommendation = shortlist[0]["candidate_file"] if shortlist else None
    return {
        "run_id": ranking["run_id"],
        "job_description": job_description,
        "scoring_profile": scoring_profile,
        "recommended_candidate": recommendation,
        "shortlist": shortlist,
    }
