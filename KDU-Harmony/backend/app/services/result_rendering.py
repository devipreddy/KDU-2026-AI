from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.phi_mapping import PhiMapping
from app.models.user import User
from app.services.context_expansion import (
    ContextualSearchResult,
    RetrievalConfidence,
    SourceCitation,
    contextual_search,
)
from app.services.cross_encoder_reranking import DEFAULT_RERANK_TOP_N, CrossEncoderReranker
from app.services.embedding_pipeline import EmbeddingEncoder
from app.services.hybrid_retrieval import DEFAULT_CANDIDATE_LIMIT, DEFAULT_RRF_K
from app.services.phi_store import (
    ROLE_VISIBILITY_FALLBACK,
    allowed_phi_entity_types_for_user,
    phi_visibility_values_from_policies,
    role_name_values,
)
from app.services.phi_tokenization import decrypt_phi_value
from app.services.query_understanding import QueryUnderstandingResult, understand_query
from app.services.retrieval_authorization import AuthorizedMetadataFilter

PHI_RENDERER_VERSION = "phi_aware_result_renderer_v1"
PHI_TOKEN_PATTERN = re.compile(r"\[[A-Z0-9_]+\]")
FULL_PHI_VISIBILITIES = {"full", "operational"}
DE_IDENTIFIED_VISIBILITIES = {"de_identified"}
METADATA_ONLY_VISIBILITIES = {"metadata_only"}
SENSITIVE_METADATA_KEYS = {
    "address",
    "dob",
    "email",
    "mrn",
    "patient_id",
    "patient_name",
    "patient_ref",
    "phone",
}

ENTITY_PLACEHOLDERS = {
    "PATIENT_NAME": "PATIENT",
    "DOB": "DOB",
    "MRN": "MRN",
    "PHONE": "PHONE",
    "ADDRESS": "ADDRESS",
    "EMAIL": "EMAIL",
}


@dataclass(frozen=True)
class PHIRenderingPolicy:
    phi_visibility: str
    render_mode: str
    can_decrypt_phi: bool
    include_clinical_text: bool
    allowed_entity_types: list[str]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "phi_visibility": self.phi_visibility,
            "render_mode": self.render_mode,
            "can_decrypt_phi": self.can_decrypt_phi,
            "include_clinical_text": self.include_clinical_text,
            "allowed_entity_types": self.allowed_entity_types,
        }


@dataclass(frozen=True)
class RenderedChunk:
    chunk_id: uuid.UUID
    section: str | None
    page_number: int | None
    start_offset: int | None
    end_offset: int | None
    token_count: int | None
    chunk_type: str | None
    text: str | None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": str(self.chunk_id),
            "section": self.section,
            "page_number": self.page_number,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "token_count": self.token_count,
            "chunk_type": self.chunk_type,
            "text": self.text,
        }


@dataclass(frozen=True)
class RenderedCitation:
    document_id: uuid.UUID
    external_id: str
    source_document: str
    source_uri: str | None
    document_type: str
    page_number: int | None
    section: str | None
    hospital: str | None
    physician: str | None
    visit_id: str | None
    checksum_sha256: str
    citation_label: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "document_id": str(self.document_id),
            "external_id": self.external_id,
            "source_document": self.source_document,
            "source_uri": self.source_uri,
            "document_type": self.document_type,
            "page_number": self.page_number,
            "section": self.section,
            "hospital": self.hospital,
            "physician": self.physician,
            "visit_id": self.visit_id,
            "checksum_sha256": self.checksum_sha256,
            "citation_label": self.citation_label,
        }


@dataclass(frozen=True)
class PHIRenderedSearchHit:
    final_rank: int
    patient_display_ref: str | None
    matched_chunk: RenderedChunk
    parent_context: RenderedChunk | None
    citation: RenderedCitation
    confidence: RetrievalConfidence
    retrieval: dict[str, Any]
    redactions: list[str]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "final_rank": self.final_rank,
            "patient_display_ref": self.patient_display_ref,
            "matched_chunk": self.matched_chunk.to_metadata(),
            "parent_context": self.parent_context.to_metadata()
            if self.parent_context is not None
            else None,
            "citation": self.citation.to_metadata(),
            "confidence": self.confidence.to_metadata(),
            "retrieval": self.retrieval,
            "redactions": self.redactions,
        }


@dataclass(frozen=True)
class PHIRenderedSearchResult:
    query: str
    authorization: AuthorizedMetadataFilter
    rendering_policy: PHIRenderingPolicy
    hits: list[PHIRenderedSearchHit]
    context_result: ContextualSearchResult

    def to_metadata(self) -> dict[str, Any]:
        return {
            "renderer": PHI_RENDERER_VERSION,
            "query": self.query,
            "authorization": self.authorization.to_metadata(),
            "rendering_policy": self.rendering_policy.to_metadata(),
            "hit_count": len(self.hits),
            "hits": [hit.to_metadata() for hit in self.hits],
        }


def phi_aware_search(
    db: Session,
    *,
    user: User,
    query: str | QueryUnderstandingResult,
    limit: int = 10,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    rerank_top_n: int = DEFAULT_RERANK_TOP_N,
    rrf_k: int = DEFAULT_RRF_K,
    authorized_patient_refs: list[str] | None = None,
    encoder: EmbeddingEncoder | None = None,
    collection: Any | None = None,
    collection_name: str | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> PHIRenderedSearchResult:
    parsed_query = understand_query(query) if isinstance(query, str) else query
    context_result = contextual_search(
        db,
        user=user,
        query=parsed_query,
        limit=limit,
        candidate_limit=candidate_limit,
        rerank_top_n=rerank_top_n,
        rrf_k=rrf_k,
        authorized_patient_refs=authorized_patient_refs,
        encoder=encoder,
        collection=collection,
        collection_name=collection_name,
        reranker=reranker,
    )
    return render_contextual_search_result(db, user=user, context_result=context_result)


def render_contextual_search_result(
    db: Session,
    *,
    user: User,
    context_result: ContextualSearchResult,
) -> PHIRenderedSearchResult:
    policy = rendering_policy_for_user(
        db,
        user=user,
        authorization=context_result.authorization,
    )
    tokens = tokens_in_context_result(context_result)
    mappings = mappings_by_token(db, tokens)
    decrypted_values = decrypted_values_by_token(
        mappings=mappings,
        policy=policy,
    )
    rendered_hits = [
        render_hit(
            hit,
            policy=policy,
            mappings=mappings,
            decrypted_values=decrypted_values,
        )
        for hit in context_result.hits
    ]
    return PHIRenderedSearchResult(
        query=context_result.query,
        authorization=context_result.authorization,
        rendering_policy=policy,
        hits=rendered_hits,
        context_result=context_result,
    )


def rendering_policy_for_user(
    db: Session,
    *,
    user: User,
    authorization: AuthorizedMetadataFilter,
) -> PHIRenderingPolicy:
    phi_visibility = authorization.phi_visibility or fallback_phi_visibility(db, user)
    role_names = role_name_values(user)
    if phi_visibility in FULL_PHI_VISIBILITIES:
        allowed_entity_types = sorted(allowed_phi_entity_types_for_user(db, user))
        return PHIRenderingPolicy(
            phi_visibility=phi_visibility,
            render_mode="full_phi",
            can_decrypt_phi=bool(allowed_entity_types),
            include_clinical_text=True,
            allowed_entity_types=allowed_entity_types,
        )
    if phi_visibility in DE_IDENTIFIED_VISIBILITIES:
        return PHIRenderingPolicy(
            phi_visibility=phi_visibility,
            render_mode="de_identified",
            can_decrypt_phi=False,
            include_clinical_text=True,
            allowed_entity_types=[],
        )
    if phi_visibility in METADATA_ONLY_VISIBILITIES or "admin" in role_names:
        return PHIRenderingPolicy(
            phi_visibility=phi_visibility,
            render_mode="metadata_only",
            can_decrypt_phi=False,
            include_clinical_text=False,
            allowed_entity_types=[],
        )
    return PHIRenderingPolicy(
        phi_visibility=phi_visibility,
        render_mode="limited",
        can_decrypt_phi=False,
        include_clinical_text=True,
        allowed_entity_types=[],
    )


def fallback_phi_visibility(db: Session, user: User) -> str:
    visibility_values = phi_visibility_values_from_policies(db, user)
    if visibility_values:
        return visibility_values[0]
    return next(
        (
            ROLE_VISIBILITY_FALLBACK[role_name]
            for role_name in role_name_values(user)
            if role_name in ROLE_VISIBILITY_FALLBACK
        ),
        "limited",
    )


def render_hit(
    hit,
    *,
    policy: PHIRenderingPolicy,
    mappings: dict[str, PhiMapping],
    decrypted_values: dict[str, str],
) -> PHIRenderedSearchHit:
    redactions: set[str] = set()
    matched_chunk = render_chunk(
        hit.matched_chunk,
        policy=policy,
        mappings=mappings,
        decrypted_values=decrypted_values,
        redactions=redactions,
    )
    parent_context = (
        render_chunk(
            hit.parent_context,
            policy=policy,
            mappings=mappings,
            decrypted_values=decrypted_values,
            redactions=redactions,
        )
        if hit.parent_context is not None
        else None
    )
    return PHIRenderedSearchHit(
        final_rank=hit.final_rank,
        patient_display_ref=patient_display_ref(
            hit.citation.document_id,
            hit.matched_chunk.chunk_id,
            policy=policy,
        ),
        matched_chunk=matched_chunk,
        parent_context=parent_context,
        citation=render_citation(
            hit.citation,
            policy=policy,
            mappings=mappings,
            decrypted_values=decrypted_values,
            redactions=redactions,
        ),
        confidence=hit.confidence,
        retrieval=render_retrieval_metadata(
            hit.retrieval,
            policy=policy,
            mappings=mappings,
            decrypted_values=decrypted_values,
            redactions=redactions,
        ),
        redactions=sorted(redactions),
    )


def render_chunk(
    chunk,
    *,
    policy: PHIRenderingPolicy,
    mappings: dict[str, PhiMapping],
    decrypted_values: dict[str, str],
    redactions: set[str],
) -> RenderedChunk:
    text = (
        render_text(
            chunk.text,
            policy=policy,
            mappings=mappings,
            decrypted_values=decrypted_values,
            redactions=redactions,
        )
        if policy.include_clinical_text
        else None
    )
    return RenderedChunk(
        chunk_id=chunk.chunk_id,
        section=chunk.section,
        page_number=chunk.page_number,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        token_count=chunk.token_count,
        chunk_type=chunk.chunk_type,
        text=text,
    )


def render_citation(
    citation: SourceCitation,
    *,
    policy: PHIRenderingPolicy,
    mappings: dict[str, PhiMapping],
    decrypted_values: dict[str, str],
    redactions: set[str],
) -> RenderedCitation:
    return RenderedCitation(
        document_id=citation.document_id,
        external_id=render_text(
            citation.external_id,
            policy=policy,
            mappings=mappings,
            decrypted_values=decrypted_values,
            redactions=redactions,
        ),
        source_document=render_text(
            citation.source_document,
            policy=policy,
            mappings=mappings,
            decrypted_values=decrypted_values,
            redactions=redactions,
        ),
        source_uri=render_text(
            citation.source_uri,
            policy=policy,
            mappings=mappings,
            decrypted_values=decrypted_values,
            redactions=redactions,
        )
        if should_include_source_uri(policy)
        else None,
        document_type=citation.document_type,
        page_number=citation.page_number,
        section=citation.section,
        hospital=citation.hospital,
        physician=citation.physician,
        visit_id=citation.visit_id,
        checksum_sha256=citation.checksum_sha256,
        citation_label=render_text(
            citation.citation_label,
            policy=policy,
            mappings=mappings,
            decrypted_values=decrypted_values,
            redactions=redactions,
        ),
    )


def should_include_source_uri(policy: PHIRenderingPolicy) -> bool:
    return policy.render_mode in {"full_phi", "metadata_only"}


def render_retrieval_metadata(
    value: Any,
    *,
    policy: PHIRenderingPolicy,
    mappings: dict[str, PhiMapping],
    decrypted_values: dict[str, str],
    redactions: set[str],
) -> dict[str, Any]:
    if policy.render_mode == "metadata_only":
        allowed_keys = {
            "sources",
            "final_rank",
            "hybrid_rank",
            "hybrid_score",
            "reranker_score",
        }
        return {key: value[key] for key in allowed_keys if isinstance(value, dict) and key in value}
    rendered = render_metadata_value(
        value,
        policy=policy,
        mappings=mappings,
        decrypted_values=decrypted_values,
        redactions=redactions,
    )
    return rendered if isinstance(rendered, dict) else {}


def render_metadata_value(
    value: Any,
    *,
    policy: PHIRenderingPolicy,
    mappings: dict[str, PhiMapping],
    decrypted_values: dict[str, str],
    redactions: set[str],
) -> Any:
    if isinstance(value, str):
        return render_text(
            value,
            policy=policy,
            mappings=mappings,
            decrypted_values=decrypted_values,
            redactions=redactions,
        )
    if isinstance(value, list):
        return [
            render_metadata_value(
                item,
                policy=policy,
                mappings=mappings,
                decrypted_values=decrypted_values,
                redactions=redactions,
            )
            for item in value
        ]
    if isinstance(value, dict):
        rendered: dict[str, Any] = {}
        for key, item in value.items():
            if key.lower() in SENSITIVE_METADATA_KEYS and policy.render_mode != "full_phi":
                rendered[key] = sensitive_metadata_placeholder(key, mode=policy.render_mode)
                redactions.add(entity_type_for_metadata_key(key))
                continue
            rendered[key] = render_metadata_value(
                item,
                policy=policy,
                mappings=mappings,
                decrypted_values=decrypted_values,
                redactions=redactions,
            )
        return rendered
    return value


def render_text(
    text: str | None,
    *,
    policy: PHIRenderingPolicy,
    mappings: dict[str, PhiMapping],
    decrypted_values: dict[str, str],
    redactions: set[str],
) -> str:
    if text is None:
        return ""

    def replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        mapping = mappings.get(token)
        entity_type = mapping.entity_type if mapping is not None else infer_entity_type(token)
        if (
            policy.can_decrypt_phi
            and mapping is not None
            and entity_type in set(policy.allowed_entity_types)
        ):
            return decrypted_values.get(token, token)

        placeholder = placeholder_for_entity(entity_type, mode=policy.render_mode)
        redactions.add(entity_type)
        return placeholder

    return PHI_TOKEN_PATTERN.sub(replace_token, text)


def placeholder_for_entity(entity_type: str, *, mode: str) -> str:
    label = ENTITY_PLACEHOLDERS.get(entity_type, "PHI")
    prefix = "DEID" if mode == "de_identified" else "REDACTED"
    return f"[{prefix}_{label}]"


def sensitive_metadata_placeholder(key: str, *, mode: str) -> str:
    return placeholder_for_entity(entity_type_for_metadata_key(key), mode=mode)


def entity_type_for_metadata_key(key: str) -> str:
    normalized = key.lower()
    if normalized in {"patient_id", "patient_name", "patient_ref"}:
        return "PATIENT_NAME"
    if normalized == "dob":
        return "DOB"
    if normalized == "mrn":
        return "MRN"
    if normalized == "phone":
        return "PHONE"
    if normalized == "address":
        return "ADDRESS"
    if normalized == "email":
        return "EMAIL"
    return "PHI"


def patient_display_ref(
    document_id: uuid.UUID,
    chunk_id: uuid.UUID,
    *,
    policy: PHIRenderingPolicy,
) -> str | None:
    if policy.render_mode == "full_phi":
        return str(document_id)
    if policy.render_mode == "metadata_only":
        return None
    prefix = "DEID" if policy.render_mode == "de_identified" else "LIMITED"
    digest = hashlib.sha256(f"{document_id}:{chunk_id}".encode()).hexdigest()[:10].upper()
    return f"{prefix}-{digest}"


def decrypted_values_by_token(
    *,
    mappings: dict[str, PhiMapping],
    policy: PHIRenderingPolicy,
) -> dict[str, str]:
    if not policy.can_decrypt_phi:
        return {}

    decrypted_at = datetime.now(UTC)
    values: dict[str, str] = {}
    allowed = set(policy.allowed_entity_types)
    for token, mapping in mappings.items():
        if mapping.entity_type not in allowed:
            continue
        values[token] = decrypt_phi_value(mapping.encrypted_value, token=token)
        mapping.last_accessed_at = decrypted_at
    return values


def mappings_by_token(db: Session, tokens: set[str]) -> dict[str, PhiMapping]:
    if not tokens:
        return {}
    mappings = db.scalars(select(PhiMapping).where(PhiMapping.token.in_(tokens))).all()
    return {mapping.token: mapping for mapping in mappings}


def tokens_in_context_result(context_result: ContextualSearchResult) -> set[str]:
    tokens: set[str] = set()
    for hit in context_result.hits:
        tokens.update(tokens_in_text(hit.matched_chunk.text))
        if hit.parent_context is not None:
            tokens.update(tokens_in_text(hit.parent_context.text))
        tokens.update(tokens_in_text(hit.citation.external_id))
        tokens.update(tokens_in_text(hit.citation.source_document))
        tokens.update(tokens_in_text(hit.citation.source_uri))
        tokens.update(tokens_in_text(hit.citation.citation_label))
        tokens.update(tokens_in_text(json.dumps(hit.retrieval, sort_keys=True)))
    return tokens


def tokens_in_text(text: str | None) -> set[str]:
    if not text:
        return set()
    return {match.group(0) for match in PHI_TOKEN_PATTERN.finditer(text)}


def infer_entity_type(token: str) -> str:
    token_body = token.strip("[]")
    if token_body.startswith("PATIENT_REF"):
        return "PATIENT_NAME"
    if token_body.startswith("DOB"):
        return "DOB"
    if token_body.startswith("MRN"):
        return "MRN"
    if token_body.startswith("PHONE"):
        return "PHONE"
    if token_body.startswith("ADDR"):
        return "ADDRESS"
    if token_body.startswith("EMAIL"):
        return "EMAIL"
    return "PHI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PHI-aware rendered search.")
    parser.add_argument("email")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT)
    parser.add_argument("--rerank-top-n", type=int, default=DEFAULT_RERANK_TOP_N)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--collection", default=settings.chroma_collection)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with SessionLocal() as db:
        user = db.scalar(
            select(User).options(selectinload(User.roles)).where(User.email == args.email)
        )
        if user is None:
            raise SystemExit(f"User not found: {args.email}")
        result = phi_aware_search(
            db,
            user=user,
            query=" ".join(args.query),
            limit=args.limit,
            candidate_limit=args.candidate_limit,
            rerank_top_n=args.rerank_top_n,
            rrf_k=args.rrf_k,
            collection_name=args.collection,
        )
    print(json.dumps(result.to_metadata(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
