"""
Retrieval and knowledge-base utilities for the Talent Intelligence RAG MCP.
"""

import json
import math
import os
import uuid
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from utils.langchain_utils import prepare_resume_documents
from utils.resume_utils import ensure_dir_exists, list_resume_files, read_resume
from utils.shortlisting_utils import build_candidate_analysis
from utils.state_utils import JsonRecordStore


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _kb_path(kb_dir):
    ensure_dir_exists(kb_dir)
    return os.path.join(os.path.abspath(kb_dir), "knowledge_base.json")


def load_knowledge_base(kb_dir):
    kb_path = _kb_path(kb_dir)
    if not os.path.exists(kb_path):
        return {"artifacts": []}

    with open(kb_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_knowledge_base(kb_dir, payload):
    kb_path = _kb_path(kb_dir)
    with open(kb_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def ingest_candidate_corpus(kb_dir, resume_dir, candidate_files=None, job_descriptions=None, notes=None):
    kb = load_knowledge_base(kb_dir)
    artifacts = kb.get("artifacts", [])
    existing_keys = {
        (
            artifact.get("artifact_type"),
            artifact.get("source"),
            artifact.get("candidate_id"),
            artifact.get("content"),
        )
        for artifact in artifacts
    }

    candidate_files = candidate_files or list_resume_files(resume_dir)
    job_descriptions = job_descriptions or []
    notes = notes or []

    for file_path in candidate_files:
        resume_text = read_resume(file_path, resume_dir)
        if not resume_text:
            continue

        dedupe_key = ("resume", file_path, os.path.splitext(os.path.basename(file_path))[0], resume_text)
        if dedupe_key in existing_keys:
            continue

        artifacts.append(
            {
                "artifact_id": str(uuid.uuid4()),
                "artifact_type": "resume",
                "source": file_path,
                "candidate_id": os.path.splitext(os.path.basename(file_path))[0],
                "content": resume_text,
                "created_at": _utc_now(),
            }
        )
        existing_keys.add(dedupe_key)

    for job_description in job_descriptions:
        dedupe_key = ("job_description", "inline", None, job_description)
        if dedupe_key in existing_keys:
            continue

        artifacts.append(
            {
                "artifact_id": str(uuid.uuid4()),
                "artifact_type": "job_description",
                "source": "inline",
                "candidate_id": None,
                "content": job_description,
                "created_at": _utc_now(),
            }
        )
        existing_keys.add(dedupe_key)

    for note in notes:
        dedupe_key = ("note", "inline", None, note)
        if dedupe_key in existing_keys:
            continue

        artifacts.append(
            {
                "artifact_id": str(uuid.uuid4()),
                "artifact_type": "note",
                "source": "inline",
                "candidate_id": None,
                "content": note,
                "created_at": _utc_now(),
            }
        )
        existing_keys.add(dedupe_key)

    payload = {"artifacts": artifacts}
    save_knowledge_base(kb_dir, payload)
    return {
        "artifact_count": len(artifacts),
        "resume_count": len([artifact for artifact in artifacts if artifact["artifact_type"] == "resume"]),
    }


def _documents_from_kb(kb_payload):
    documents = []
    for artifact in kb_payload.get("artifacts", []):
        if artifact["artifact_type"] == "resume":
            chunks = prepare_resume_documents(
                artifact["content"],
                artifact["source"],
            )["chunks"]
            for chunk in chunks:
                metadata = dict(chunk.metadata)
                metadata.update(
                    {
                        "artifact_id": artifact["artifact_id"],
                        "artifact_type": artifact["artifact_type"],
                        "candidate_id": artifact["candidate_id"],
                        "source": artifact["source"],
                    }
                )
                documents.append(Document(page_content=chunk.page_content, metadata=metadata))
        else:
            documents.append(
                Document(
                    page_content=artifact["content"],
                    metadata={
                        "artifact_id": artifact["artifact_id"],
                        "artifact_type": artifact["artifact_type"],
                        "candidate_id": artifact.get("candidate_id"),
                        "source": artifact["source"],
                    },
                )
            )
    return documents


def _lexical_retrieval(query, documents, top_k=5):
    query_terms = set(query.lower().split())
    scored = []
    for document in documents:
        doc_terms = set(document.page_content.lower().split())
        overlap = len(query_terms & doc_terms)
        score = overlap / max(1, len(query_terms))
        scored.append((document, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def _retrieve_documents(query, documents, embeddings, top_k=5):
    if embeddings and documents:
        vectorstore = FAISS.from_documents(documents, embeddings)
        retrieved = vectorstore.similarity_search_with_score(query, k=min(top_k, len(documents)))
        return [(document, 1 / (1 + score)) for document, score in retrieved]

    return _lexical_retrieval(query, documents, top_k=top_k)


def query_talent_knowledge_base(kb_dir, query, embeddings, llm, top_k=5):
    kb_payload = load_knowledge_base(kb_dir)
    documents = _documents_from_kb(kb_payload)
    if not documents:
        raise ValueError("Knowledge base is empty. Ingest candidate artifacts first.")

    retrieved = _retrieve_documents(query, documents, embeddings, top_k=top_k)
    source_summaries = [
        {
            "source": document.metadata.get("source"),
            "artifact_type": document.metadata.get("artifact_type"),
            "candidate_id": document.metadata.get("candidate_id"),
            "score": round(score * 100, 2),
            "excerpt": " ".join(document.page_content.split())[:500],
        }
        for document, score in retrieved
    ]

    if llm:
        context = "\n\n".join(
            f"Source: {item['source']}\nType: {item['artifact_type']}\nExcerpt: {item['excerpt']}"
            for item in source_summaries
        )
        prompt = f"""
        You are a hiring intelligence assistant.

        Answer the query using only the retrieved evidence below.
        If the evidence is insufficient, say so clearly.

        Query:
        {query}

        Retrieved evidence:
        {context}

        Return:
        1. Answer
        2. Confidence
        3. Follow-up recommendation
        """
        answer = llm.invoke(prompt).content
    else:
        answer = (
            "Retrieved evidence is available, but no LLM is configured. "
            "Use the cited excerpts to inspect the talent corpus manually."
        )

    return {
        "query_id": str(uuid.uuid4()),
        "query": query,
        "answer": answer,
        "sources": source_summaries,
    }


def persist_query_result(query_store_dir, query_payload):
    store = JsonRecordStore(query_store_dir)
    return store.create(query_payload, record_id=query_payload["query_id"])


def load_query_result(query_store_dir, query_id):
    store = JsonRecordStore(query_store_dir)
    return store.load(query_id)


def _cosine_similarity(left, right):
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def find_similar_candidates(kb_dir, candidate_file, resume_dir, embeddings, top_k=3):
    kb_payload = load_knowledge_base(kb_dir)
    resume_artifacts = [artifact for artifact in kb_payload.get("artifacts", []) if artifact["artifact_type"] == "resume"]
    if not resume_artifacts:
        raise ValueError("Knowledge base has no ingested resume artifacts.")

    target_candidate_id = os.path.splitext(os.path.basename(candidate_file))[0]
    target_artifact = next((artifact for artifact in resume_artifacts if artifact["candidate_id"] == target_candidate_id), None)
    if not target_artifact:
        resume_text = read_resume(candidate_file, resume_dir)
        if not resume_text:
            raise ValueError(f"Failed to read resume: {candidate_file}")
        target_artifact = {
            "candidate_id": target_candidate_id,
            "source": candidate_file,
            "content": resume_text,
        }

    candidates = [artifact for artifact in resume_artifacts if artifact["candidate_id"] != target_candidate_id]

    if embeddings:
        vectors = embeddings.embed_documents([target_artifact["content"]] + [artifact["content"] for artifact in candidates])
        target_vector = vectors[0]
        similar = []
        for artifact, vector in zip(candidates, vectors[1:]):
            similar.append(
                {
                    "candidate_file": artifact["source"],
                    "candidate_id": artifact["candidate_id"],
                    "similarity": round(_cosine_similarity(target_vector, vector) * 100, 2),
                }
            )
        similar.sort(key=lambda item: item["similarity"], reverse=True)
        return similar[:top_k]

    target_terms = set(target_artifact["content"].lower().split())
    similar = []
    for artifact in candidates:
        artifact_terms = set(artifact["content"].lower().split())
        overlap = len(target_terms & artifact_terms) / max(1, len(target_terms))
        similar.append(
            {
                "candidate_file": artifact["source"],
                "candidate_id": artifact["candidate_id"],
                "similarity": round(overlap * 100, 2),
            }
        )
    similar.sort(key=lambda item: item["similarity"], reverse=True)
    return similar[:top_k]


def generate_gap_analysis(candidate_file, job_description, resume_dir, embeddings, llm):
    analysis = build_candidate_analysis(
        file_path=candidate_file,
        resume_dir=resume_dir,
        job_description=job_description,
        embeddings=embeddings,
        llm=llm,
    )

    gaps = []
    for keyword in job_description.split():
        normalized = keyword.lower().strip(",.()")
        if normalized and normalized not in [item.lower() for item in analysis["matched_keywords"]]:
            gaps.append(normalized)

    return {
        "candidate_file": candidate_file,
        "overall_score": analysis["overall_score"],
        "matched_keywords": analysis["matched_keywords"],
        "gap_keywords": sorted(set(gaps))[:20],
        "assessment": analysis["assessment"],
        "evidence": analysis["evidence"],
    }


def generate_interview_questions(candidate_file, job_description, resume_dir, embeddings, llm, question_count=5):
    analysis = build_candidate_analysis(
        file_path=candidate_file,
        resume_dir=resume_dir,
        job_description=job_description,
        embeddings=embeddings,
        llm=llm,
    )

    if llm:
        prompt = f"""
        Generate {question_count} targeted interview questions for this candidate.

        Candidate summary:
        {json.dumps(analysis, indent=2)}

        Job description:
        {job_description}

        Ensure the questions test both strengths and missing areas.
        """
        generated = llm.invoke(prompt).content
    else:
        generated = "\n".join(
            [
                f"{index}. Tell us about a project where you used {keyword}."
                for index, keyword in enumerate(analysis["matched_keywords"][:question_count], start=1)
            ]
        )

    return {
        "candidate_file": candidate_file,
        "questions": generated,
        "evidence": analysis["evidence"],
    }


def explain_answer_sources(query_store_dir, query_id):
    query_payload = load_query_result(query_store_dir, query_id)
    explanations = []
    for index, source in enumerate(query_payload.get("sources", []), start=1):
        explanations.append(
            {
                "source_rank": index,
                "source": source["source"],
                "artifact_type": source["artifact_type"],
                "candidate_id": source.get("candidate_id"),
                "score": source["score"],
                "explanation": (
                    f"Source {index} contributed because it is a {source['artifact_type']} artifact "
                    f"with similarity score {source['score']} and content overlapping the query intent."
                ),
                "excerpt": source["excerpt"],
            }
        )

    return {
        "query_id": query_id,
        "query": query_payload.get("query"),
        "source_explanations": explanations,
    }
