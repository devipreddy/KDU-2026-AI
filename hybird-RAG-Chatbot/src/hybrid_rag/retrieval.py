from __future__ import annotations

import logging
from typing import Any

from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .config import Settings
from .utils import content_hash, cosine_similarity_from_tokens, keyword_tokenize, sanitize_context


logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, settings: Settings, vectorstore: Chroma, registry: list[dict[str, Any]]) -> None:
        self.settings = settings
        self.vectorstore = vectorstore
        self.registry = registry
        self._bm25 = self._build_bm25(registry)
        self._reranker: CrossEncoder | None = None
        self._reranker_error: str | None = None

    def refresh_registry(self, registry: list[dict[str, Any]]) -> None:
        self.registry = registry
        self._bm25 = self._build_bm25(registry)

    def _build_bm25(self, registry: list[dict[str, Any]]) -> BM25Okapi | None:
        if not registry:
            return None
        tokenized = [keyword_tokenize(item["text"]) for item in registry]
        return BM25Okapi(tokenized)

    def semantic_search(self, query: str) -> list[dict[str, Any]]:
        candidates = self.vectorstore.max_marginal_relevance_search(
            query=query,
            k=self.settings.top_k_semantic,
            fetch_k=self.settings.mmr_fetch_k,
            lambda_mult=self.settings.mmr_lambda,
        )
        embedding_model = self.vectorstore._embedding_function
        query_embedding = embedding_model.embed_query(query)
        candidate_embeddings = embedding_model.embed_documents(
            [doc.page_content for doc in candidates]
        )
        docs: list[dict[str, Any]] = []
        for doc, embedding in zip(candidates, candidate_embeddings):
            score = self._embedding_cosine_similarity(query_embedding, embedding)
            docs.append(
                {
                    "id": doc.metadata.get("chunk_id", content_hash(doc.page_content)),
                    "page_content": sanitize_context(doc.page_content),
                    "metadata": dict(doc.metadata),
                    "semantic_score": float(score),
                    "keyword_score": 0.0,
                }
            )
        return docs

    def keyword_search(self, query: str) -> list[dict[str, Any]]:
        if not self._bm25 or not self.registry:
            return []
        scores = self._bm25.get_scores(keyword_tokenize(query))
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )[: self.settings.top_k_keyword]
        docs: list[dict[str, Any]] = []
        for idx in ranked_indices:
            item = self.registry[idx]
            docs.append(
                {
                    "id": item["chunk_id"],
                    "page_content": sanitize_context(item["text"]),
                    "metadata": item["metadata"],
                    "semantic_score": 0.0,
                    "keyword_score": float(scores[idx]),
                }
            )
        return docs

    def hybrid_search(self, query: str) -> list[dict[str, Any]]:
        return self.semantic_search(query) + self.keyword_search(query)

    def deduplicate(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for doc in docs:
            doc_id = doc["id"]
            if doc_id not in merged:
                merged[doc_id] = doc
                continue
            merged[doc_id]["semantic_score"] = max(merged[doc_id]["semantic_score"], doc["semantic_score"])
            merged[doc_id]["keyword_score"] = max(merged[doc_id]["keyword_score"], doc["keyword_score"])
        exact_deduped = list(merged.values())
        filtered: list[dict[str, Any]] = []
        for doc in exact_deduped:
            is_duplicate = False
            for kept in filtered:
                similarity = cosine_similarity_from_tokens(doc["page_content"], kept["page_content"])
                if similarity >= self.settings.dedup_similarity_threshold:
                    is_duplicate = True
                    if doc.get("semantic_score", 0.0) + doc.get("keyword_score", 0.0) > kept.get(
                        "semantic_score", 0.0
                    ) + kept.get("keyword_score", 0.0):
                        filtered.remove(kept)
                        filtered.append(doc)
                    break
            if not is_duplicate:
                filtered.append(doc)
        return filtered

    def rerank(self, query: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not docs:
            return []
        reranker = self._get_reranker()
        pairs = [(query, doc["page_content"]) for doc in docs]
        scores = reranker.predict(pairs) if reranker else [0.0] * len(docs)
        ranked = []
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
            doc["context_coverage"] = self._context_coverage(query, doc["page_content"])
            doc["combined_score"] = (
                (0.45 * doc.get("semantic_score", 0.0))
                + (0.15 * doc.get("keyword_score", 0.0))
                + (0.25 * doc["rerank_score"])
                + (0.15 * doc["context_coverage"])
            )
            ranked.append(doc)
        return sorted(ranked, key=lambda item: item["combined_score"], reverse=True)

    def _get_reranker(self) -> CrossEncoder | None:
        if self._reranker is not None:
            return self._reranker
        if self._reranker_error is not None:
            return None
        try:
            self._reranker = CrossEncoder(self.settings.reranker_model)
            return self._reranker
        except Exception as exc:  # pragma: no cover - runtime dependency fallback
            self._reranker_error = str(exc)
            logger.warning("Cross-encoder reranker unavailable, falling back to retrieval-only ranking: %s", exc)
            return None

    def get_reranker_status(self) -> dict[str, Any]:
        return {
            "available": self._reranker is not None and self._reranker_error is None,
            "error": self._reranker_error,
            "model": self.settings.reranker_model,
        }

    def filter_relevant(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filtered = [
            doc
            for doc in docs
            if doc.get("rerank_score", 0.0) >= self.settings.rerank_min_score
            or doc.get("context_coverage", 0.0) >= self.settings.min_context_coverage
        ]
        return filtered or docs[: self.settings.top_k_final]

    def select_top_k(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        for doc in docs:
            source = doc["metadata"].get("source", "unknown")
            if len(selected) >= self.settings.top_k_final:
                break
            if source in seen_sources and len(docs) > self.settings.top_k_final:
                continue
            selected.append(doc)
            seen_sources.add(source)
        if len(selected) < self.settings.top_k_final:
            for doc in docs:
                if doc not in selected:
                    selected.append(doc)
                if len(selected) >= self.settings.top_k_final:
                    break
        return selected

    def _context_coverage(self, query: str, text: str) -> float:
        query_tokens = set(keyword_tokenize(query))
        if not query_tokens:
            return 0.0
        text_tokens = set(keyword_tokenize(text))
        overlap = query_tokens.intersection(text_tokens)
        return len(overlap) / max(len(query_tokens), 1)

    def _embedding_cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)
