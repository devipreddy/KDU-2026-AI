from __future__ import annotations

import argparse
import json
import math
import re
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.session import SessionLocal
from app.models.document_chunk import DocumentChunk
from app.models.phi_mapping import PhiMapping
from app.models.user import User
from app.services.chroma_index import build_bm25_document
from app.services.clinical_metadata import MEDICATION_TERMS, match_terms, ordered_unique
from app.services.phi_tokenization import canonicalize_phi_value, decrypt_phi_value
from app.services.query_understanding import QueryUnderstandingResult, understand_query
from app.services.retrieval_authorization import (
    AuthorizedMetadataFilter,
    build_authorized_metadata_filter,
)

BM25_RETRIEVER_VERSION = "local_bm25_retriever_v1"
BM25_K1 = 1.5
BM25_B = 0.75
EXACT_FIELD_BOOST = 4.0
MRN_EXACT_BOOST = 8.0
CONTENT_TERM_BOOST = 1.15
DIRECT_IDENTIFIER_VISIBILITIES = {"full", "operational"}
MRN_QUERY_PATTERN = re.compile(
    r"\b(?:MRN|medical\s+record(?:\s+number)?|record\s+number)\s*[:#-]?\s*"
    r"(?P<value>[A-Z]{0,6}-?\d{4,}(?:-\d+)?)\b",
    flags=re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"\[[A-Z0-9_]+\]|[A-Z0-9]+(?:[._-][A-Z0-9]+)*", flags=re.IGNORECASE)


@dataclass(frozen=True)
class BM25QueryTerms:
    tokens: list[str]
    medication_names: list[str]
    icd_codes: list[str]
    mrns: list[str]
    mrn_tokens: list[str]
    mrn_patient_refs: list[str]
    physician_names: list[str]
    diagnosis_terms: list[str]
    fallback_terms: list[str]

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BM25SearchHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    patient_ref: str
    section: str | None
    score: float
    bm25_score: float
    exact_match_score: float
    matched_fields: dict[str, list[str]]
    snippet: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "patient_ref": self.patient_ref,
            "section": self.section,
            "score": self.score,
            "bm25_score": self.bm25_score,
            "exact_match_score": self.exact_match_score,
            "matched_fields": self.matched_fields,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class BM25SearchResult:
    query: str
    query_terms: BM25QueryTerms
    authorization: AuthorizedMetadataFilter
    hits: list[BM25SearchHit]
    candidate_chunk_count: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "retriever": BM25_RETRIEVER_VERSION,
            "query": self.query,
            "query_terms": self.query_terms.to_metadata(),
            "authorization": self.authorization.to_metadata(),
            "candidate_chunk_count": self.candidate_chunk_count,
            "hits": [hit.to_metadata() for hit in self.hits],
        }


def bm25_search(
    db: Session,
    *,
    user: User,
    query: str | QueryUnderstandingResult,
    limit: int = 10,
    authorized_patient_refs: list[str] | None = None,
    authorization: AuthorizedMetadataFilter | None = None,
) -> BM25SearchResult:
    parsed_query = understand_query(query) if isinstance(query, str) else query
    raw_query = parsed_query.original_query
    authorization = authorization or build_authorized_metadata_filter(
        db,
        user=user,
        query=parsed_query,
        authorized_patient_refs=authorized_patient_refs,
    )
    query_terms = extract_bm25_query_terms(
        db,
        query=parsed_query,
        authorization=authorization,
    )

    if authorization.denied or not query_terms.tokens:
        return BM25SearchResult(
            query=raw_query,
            query_terms=query_terms,
            authorization=authorization,
            hits=[],
            candidate_chunk_count=0,
        )

    chunks = authorized_candidate_chunks(db, authorization=authorization, query_terms=query_terms)
    hits = score_bm25_chunks(chunks, query_terms=query_terms, limit=limit)
    return BM25SearchResult(
        query=raw_query,
        query_terms=query_terms,
        authorization=authorization,
        hits=hits,
        candidate_chunk_count=len(chunks),
    )


def extract_bm25_query_terms(
    db: Session,
    *,
    query: QueryUnderstandingResult,
    authorization: AuthorizedMetadataFilter,
) -> BM25QueryTerms:
    medication_names = match_terms(query.original_query, MEDICATION_TERMS)
    icd_codes = query.icd_codes
    physician_names = query.physicians
    diagnosis_terms = ordered_unique(
        diagnosis for concept in query.diagnosis_concepts for diagnosis in concept.diagnoses
    )
    mrns = extract_mrn_values(query.original_query)
    resolved_mrns = resolve_mrn_search_terms(
        db,
        mrns=mrns,
        authorization=authorization,
    )
    fallback_terms = fallback_query_terms(query.original_query)

    exact_terms = [
        *medication_names,
        *icd_codes,
        *resolved_mrns["tokens"],
        *physician_names,
        *diagnosis_terms,
    ]
    tokens = ordered_unique(
        token for term in [*exact_terms, *fallback_terms] for token in tokenize_lexical_text(term)
    )
    return BM25QueryTerms(
        tokens=tokens,
        medication_names=medication_names,
        icd_codes=icd_codes,
        mrns=mrns,
        mrn_tokens=resolved_mrns["tokens"],
        mrn_patient_refs=resolved_mrns["patient_refs"],
        physician_names=physician_names,
        diagnosis_terms=diagnosis_terms,
        fallback_terms=fallback_terms,
    )


def resolve_mrn_search_terms(
    db: Session,
    *,
    mrns: list[str],
    authorization: AuthorizedMetadataFilter,
) -> dict[str, list[str]]:
    if not mrns or authorization.phi_visibility not in DIRECT_IDENTIFIER_VISIBILITIES:
        return {"tokens": [], "patient_refs": []}

    canonical_mrns = {canonicalize_phi_value("MRN", mrn) for mrn in mrns}
    tokens: list[str] = []
    patient_refs: list[str] = []
    mappings = db.scalars(select(PhiMapping).where(PhiMapping.entity_type == "MRN")).all()
    for mapping in mappings:
        value = decrypt_phi_value(mapping.encrypted_value, token=mapping.token)
        stored_canonical = canonicalize_phi_value("MRN", value)
        if not any(mrn_values_match(query_mrn, stored_canonical) for query_mrn in canonical_mrns):
            continue
        tokens.append(mapping.token)
        patient_refs.append(mapping.patient_ref)
    return {
        "tokens": ordered_unique(tokens),
        "patient_refs": ordered_unique(patient_refs),
    }


def mrn_values_match(query_mrn: str, stored_mrn: str) -> bool:
    if query_mrn == stored_mrn:
        return True
    query_digits = re.sub(r"\D+", "", query_mrn)
    stored_digits = re.sub(r"\D+", "", stored_mrn)
    return bool(query_digits and query_digits == stored_digits)


def authorized_candidate_chunks(
    db: Session,
    *,
    authorization: AuthorizedMetadataFilter,
    query_terms: BM25QueryTerms,
) -> list[DocumentChunk]:
    chunks = db.scalars(
        select(DocumentChunk)
        .options(selectinload(DocumentChunk.document))
        .order_by(DocumentChunk.created_at, DocumentChunk.chunk_index)
    ).all()

    candidates = [
        chunk for chunk in chunks if chunk_matches_where(chunk, authorization.chroma_where)
    ]
    if query_terms.mrn_patient_refs:
        allowed_refs = set(query_terms.mrn_patient_refs)
        candidates = [chunk for chunk in candidates if chunk_patient_ref(chunk) in allowed_refs]
    return candidates


def score_bm25_chunks(
    chunks: list[DocumentChunk],
    *,
    query_terms: BM25QueryTerms,
    limit: int,
) -> list[BM25SearchHit]:
    corpus_tokens = [tokenize_lexical_text(bm25_search_text(chunk)) for chunk in chunks]
    bm25_scores = bm25_scores_for_corpus(corpus_tokens, query_terms.tokens)

    hits: list[BM25SearchHit] = []
    for chunk, bm25_score in zip(chunks, bm25_scores, strict=False):
        matched_fields = exact_matched_fields(chunk, query_terms)
        exact_score = exact_match_score(matched_fields)
        total_score = round(bm25_score + exact_score, 6)
        if total_score <= 0:
            continue
        hits.append(
            BM25SearchHit(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                patient_ref=chunk_patient_ref(chunk),
                section=chunk.section,
                score=total_score,
                bm25_score=round(bm25_score, 6),
                exact_match_score=round(exact_score, 6),
                matched_fields=matched_fields,
                snippet=build_snippet(chunk.content, query_terms),
            )
        )

    return sorted(
        hits,
        key=lambda hit: (-hit.score, str(hit.document_id), hit.section or "", str(hit.chunk_id)),
    )[:limit]


def bm25_scores_for_corpus(
    corpus_tokens: list[list[str]],
    query_tokens: list[str],
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[float]:
    if not corpus_tokens or not query_tokens:
        return []

    document_count = len(corpus_tokens)
    document_lengths = [len(tokens) for tokens in corpus_tokens]
    average_document_length = sum(document_lengths) / document_count if document_count else 0
    document_frequency: Counter[str] = Counter()
    token_counts = [Counter(tokens) for tokens in corpus_tokens]

    for tokens in corpus_tokens:
        document_frequency.update(set(tokens))

    scores: list[float] = []
    for tokens, counts, document_length in zip(
        corpus_tokens,
        token_counts,
        document_lengths,
        strict=False,
    ):
        if not tokens:
            scores.append(0.0)
            continue

        score = 0.0
        for query_token in query_tokens:
            term_frequency = counts.get(query_token, 0)
            if term_frequency == 0:
                continue
            df = document_frequency[query_token]
            idf = math.log(1 + ((document_count - df + 0.5) / (df + 0.5)))
            denominator = term_frequency + k1 * (
                1 - b + b * (document_length / average_document_length)
            )
            score += idf * ((term_frequency * (k1 + 1)) / denominator)
        scores.append(score)
    return scores


def exact_matched_fields(
    chunk: DocumentChunk,
    query_terms: BM25QueryTerms,
) -> dict[str, list[str]]:
    field_sources = exact_field_sources(chunk)
    matched: dict[str, list[str]] = {}

    for field_name, terms in (
        ("medication_names", query_terms.medication_names),
        ("icd_codes", query_terms.icd_codes),
        ("mrn_tokens", query_terms.mrn_tokens),
        ("physician_names", query_terms.physician_names),
        ("diagnosis_terms", query_terms.diagnosis_terms),
        ("content_terms", query_terms.fallback_terms),
    ):
        source_text = field_sources[field_name]
        field_matches = [term for term in terms if contains_exact_term(source_text, term)]
        if field_matches:
            matched[field_name] = ordered_unique(field_matches)
    return matched


def exact_field_sources(chunk: DocumentChunk) -> dict[str, str]:
    metadata = chunk.retrieval_metadata or {}
    clinical_entities = metadata.get("clinical_entities") or {}
    return {
        "medication_names": join_source_values(
            chunk.content,
            clinical_entities.get("medications") if isinstance(clinical_entities, dict) else [],
        ),
        "icd_codes": join_source_values(
            chunk.content,
            metadata.get("icd_codes"),
            clinical_entities.get("icd_codes") if isinstance(clinical_entities, dict) else [],
        ),
        "mrn_tokens": chunk.content,
        "physician_names": join_source_values(chunk.content, metadata.get("physician")),
        "diagnosis_terms": join_source_values(
            chunk.content,
            metadata.get("diagnosis"),
            clinical_entities.get("diagnoses") if isinstance(clinical_entities, dict) else [],
        ),
        "content_terms": bm25_search_text(chunk),
    }


def exact_match_score(matched_fields: dict[str, list[str]]) -> float:
    score = 0.0
    for field_name, terms in matched_fields.items():
        if field_name == "mrn_tokens":
            field_boost = MRN_EXACT_BOOST
        elif field_name == "content_terms":
            field_boost = CONTENT_TERM_BOOST
        else:
            field_boost = EXACT_FIELD_BOOST
        score += field_boost * len(terms)
    return score


def bm25_search_text(chunk: DocumentChunk) -> str:
    metadata = chunk.retrieval_metadata or {}
    clinical_entities = metadata.get("clinical_entities") or {}
    extra_parts = [
        metadata.get("diagnosis"),
        metadata.get("physician"),
        metadata.get("hospital"),
        metadata.get("document_type"),
        metadata.get("icd_codes"),
    ]
    if isinstance(clinical_entities, dict):
        extra_parts.extend(
            [
                clinical_entities.get("medications"),
                clinical_entities.get("icd_codes"),
                clinical_entities.get("diagnoses"),
                clinical_entities.get("symptoms"),
            ]
        )
    return join_source_values(build_bm25_document(chunk), *extra_parts)


def chunk_matches_where(chunk: DocumentChunk, where: dict[str, Any]) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(chunk_matches_where(chunk, predicate) for predicate in where["$and"])
    if "$or" in where:
        return any(chunk_matches_where(chunk, predicate) for predicate in where["$or"])

    for field, condition in where.items():
        value = chunk_filter_value(chunk, field)
        if not condition_matches(value, condition):
            return False
    return True


def condition_matches(value: Any, condition: Any) -> bool:
    values = normalize_filter_values(value)
    if isinstance(condition, dict):
        for operator, expected in condition.items():
            if operator == "$eq" and not values_equal(values, [expected]):
                return False
            if operator == "$in" and not values_intersect(
                values, normalize_filter_values(expected)
            ):
                return False
            if operator == "$gte" and not values_compare(values, str(expected), ">="):
                return False
            if operator == "$lte" and not values_compare(values, str(expected), "<="):
                return False
        return True
    return values_equal(values, [condition])


def chunk_filter_value(chunk: DocumentChunk, field: str) -> Any:
    metadata = chunk.retrieval_metadata or {}
    clinical_entities = metadata.get("clinical_entities") or {}
    if field == "sensitivity_level":
        return chunk.sensitivity_level.value
    if field == "document_id":
        return str(chunk.document_id)
    if field == "chunk_id":
        return str(chunk.id)
    if field == "patient_ref":
        return chunk_patient_ref(chunk)
    if field == "icd_codes":
        return metadata.get("icd_codes") or (
            clinical_entities.get("icd_codes") if isinstance(clinical_entities, dict) else []
        )
    if field == "visit_date":
        dates = clinical_entities.get("dates") if isinstance(clinical_entities, dict) else []
        return first_visit_date_value(dates)
    return metadata.get(field, "")


def first_visit_date_value(dates: Any) -> str:
    if not isinstance(dates, list):
        return ""
    for date_entry in dates:
        if isinstance(date_entry, dict) and date_entry.get("value"):
            return str(date_entry["value"])
    return ""


def normalize_filter_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set | frozenset):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def values_equal(values: list[str], expected_values: list[Any]) -> bool:
    expected = {str(value) for value in expected_values}
    return any(value in expected for value in values)


def values_intersect(values: list[str], expected_values: list[str]) -> bool:
    return bool(set(values).intersection(expected_values))


def values_compare(values: list[str], expected: str, operator: str) -> bool:
    if operator == ">=":
        return any(value >= expected for value in values)
    return any(value <= expected for value in values)


def chunk_patient_ref(chunk: DocumentChunk) -> str:
    return str((chunk.retrieval_metadata or {}).get("patient_ref") or chunk.document.patient_ref)


def tokenize_lexical_text(text: str) -> list[str]:
    base_tokens = [normalize_token(match.group(0)) for match in TOKEN_PATTERN.finditer(text)]
    tokens: list[str] = []
    for token in base_tokens:
        tokens.append(token)
        tokens.extend(token_variants(token))

    for left, right in zip(base_tokens, base_tokens[1:], strict=False):
        if is_compoundable_token(left) and is_compoundable_token(right):
            tokens.append(f"{left}{right}")
    return tokens


def normalize_token(token: str) -> str:
    return token.lower()


def token_variants(token: str) -> list[str]:
    if is_phi_token(token):
        return []

    variants: list[str] = []
    collapsed = re.sub(r"[._-]+", "", token)
    if collapsed != token and len(collapsed) > 2:
        variants.append(collapsed)

    singular = singularize_token(token)
    if singular and singular != token:
        variants.append(singular)
    return ordered_unique(variants)


def singularize_token(token: str) -> str | None:
    if not token.isalpha():
        return None
    if len(token) <= 3:
        return None
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith(("ss", "us", "is", "sis", "tes")):
        return None
    if token.endswith("s"):
        return token[:-1]
    return None


def is_compoundable_token(token: str) -> bool:
    return len(token) >= 5 and token.isalpha()


def is_phi_token(token: str) -> bool:
    return token.startswith("[") and token.endswith("]")


def contains_exact_term(text: str, term: str) -> bool:
    text_tokens = set(tokenize_lexical_text(text))
    term_tokens = [normalize_token(match.group(0)) for match in TOKEN_PATTERN.finditer(term)]
    if not term_tokens:
        return False

    return all(
        any(candidate in text_tokens for candidate in [token, *token_variants(token)])
        for token in term_tokens
    )


def extract_mrn_values(query: str) -> list[str]:
    return ordered_unique(match.group("value") for match in MRN_QUERY_PATTERN.finditer(query))


def fallback_query_terms(query: str) -> list[str]:
    stopwords = {
        "and",
        "by",
        "for",
        "from",
        "in",
        "of",
        "patient",
        "patients",
        "record",
        "records",
        "show",
        "the",
        "with",
    }
    return [
        token for token in tokenize_lexical_text(query) if len(token) > 2 and token not in stopwords
    ]


def join_source_values(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            parts.append(value)
            continue
        if isinstance(value, list | tuple | set | frozenset):
            parts.extend(str(item) for item in value if item is not None)
            continue
        parts.append(str(value))
    return "\n".join(part for part in parts if part)


def build_snippet(content: str, query_terms: BM25QueryTerms, *, max_length: int = 220) -> str:
    tokens = query_terms.tokens
    lower_content = content.lower()
    first_match = next(
        (lower_content.find(token) for token in tokens if lower_content.find(token) >= 0),
        -1,
    )
    if first_match < 0:
        return content[:max_length].strip()

    start = max(0, first_match - 60)
    end = min(len(content), start + max_length)
    return content[start:end].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run authorization-aware BM25 retrieval.")
    parser.add_argument("email")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with SessionLocal() as db:
        user = db.scalar(
            select(User).options(selectinload(User.roles)).where(User.email == args.email)
        )
        if user is None:
            raise SystemExit(f"User not found: {args.email}")
        result = bm25_search(
            db,
            user=user,
            query=" ".join(args.query),
            limit=args.limit,
        )
    print(json.dumps(result.to_metadata(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
