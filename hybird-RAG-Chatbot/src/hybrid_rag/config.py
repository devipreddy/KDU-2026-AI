from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str
    openrouter_embedding_model: str
    openrouter_base_url: str
    openrouter_app_name: str
    openrouter_site_url: str
    chroma_dir: str
    docstore_path: str
    session_store_path: str
    trace_dir: str
    collection_name: str = "hybrid_rag_docs"
    chunk_size: int = 1200
    chunk_overlap: int = 220
    top_k_semantic: int = 8
    top_k_keyword: int = 8
    top_k_final: int = 4
    mmr_fetch_k: int = 20
    mmr_lambda: float = 0.65
    max_iterations: int = 2
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    confidence_threshold: float = 0.75
    rerank_min_score: float = 0.2
    min_context_coverage: float = 0.3
    dedup_similarity_threshold: float = 0.92
    input_token_cost_per_1k: float = 0.0
    output_token_cost_per_1k: float = 0.0
    llm_max_tokens: int = 1200
    llm_analysis_max_tokens: int = 300
    llm_summary_max_tokens: int = 250


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini"),
        openrouter_embedding_model=os.getenv(
            "OPENROUTER_EMBEDDING_MODEL",
            "text-embedding-3-small",
        ),
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ),
        openrouter_app_name=os.getenv(
            "OPENROUTER_APP_NAME",
            "Hybrid Search RAG Chatbot",
        ),
        openrouter_site_url=os.getenv(
            "OPENROUTER_SITE_URL",
            "https://localhost",
        ),
        chroma_dir=os.getenv("CHROMA_DIR", "./storage/chroma"),
        docstore_path=os.getenv("DOCSTORE_PATH", "./storage/docstore.json"),
        session_store_path=os.getenv("SESSION_STORE_PATH", "./storage/sessions.json"),
        trace_dir=os.getenv("TRACE_DIR", "./storage/traces"),
        mmr_fetch_k=int(os.getenv("MMR_FETCH_K", "20")),
        mmr_lambda=float(os.getenv("MMR_LAMBDA", "0.65")),
        max_iterations=int(os.getenv("MAX_ITERATIONS", "2")),
        top_k_final=int(os.getenv("TOP_K_FINAL", "4")),
        rerank_min_score=float(os.getenv("RERANK_MIN_SCORE", "0.2")),
        min_context_coverage=float(os.getenv("MIN_CONTEXT_COVERAGE", "0.3")),
        dedup_similarity_threshold=float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.92")),
        input_token_cost_per_1k=float(os.getenv("INPUT_TOKEN_COST_PER_1K", "0")),
        output_token_cost_per_1k=float(os.getenv("OUTPUT_TOKEN_COST_PER_1K", "0")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1200")),
        llm_analysis_max_tokens=int(os.getenv("LLM_ANALYSIS_MAX_TOKENS", "300")),
        llm_summary_max_tokens=int(os.getenv("LLM_SUMMARY_MAX_TOKENS", "250")),
    )
