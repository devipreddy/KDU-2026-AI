from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

import pytest

import auto_design.catalog.retrieval as retrieval_module
from auto_design.catalog.retrieval import (
    AsyncCatalogRetriever,
    CatalogRetrievalQuery,
    RetrievalCacheKey,
    UnknownCatalogPartitionError,
)
from auto_design.catalog.service import CatalogService


ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "catalog.json"


def build_retriever() -> AsyncCatalogRetriever:
    return AsyncCatalogRetriever(CatalogService.load(CATALOG_PATH))


def test_retrieve_category_filters_by_style_color_and_caches_semantic_key() -> None:
    retriever = build_retriever()
    first_query = CatalogRetrievalQuery(
        category="cabinets",
        style="modern",
        color="#1f3a5f",
    )
    second_query = CatalogRetrievalQuery(
        category="cabinets",
        style="MODERN",
        color="#1F3A5F",
        query="corner",
    )

    first = asyncio.run(retriever.retrieve_category(first_query))
    second = asyncio.run(retriever.retrieve_category(second_query))

    assert [product.id for product in first.products] == ["SKU-C11"]
    assert [product.id for product in second.products] == ["SKU-C11"]
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert retriever.cache_hits == 1
    assert retriever.cache_misses == 1
    assert retriever.cache_keys == (
        RetrievalCacheKey(category="cabinets", style="modern", color="#1F3A5F"),
    )


def test_retrieve_many_uses_asyncio_gather_for_parallel_category_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = build_retriever()
    original_gather = retrieval_module.asyncio.gather
    gather_call_sizes: list[int] = []

    async def spy_gather(*aws: Awaitable[Any]) -> list[Any]:
        gather_call_sizes.append(len(aws))
        return await original_gather(*aws)

    monkeypatch.setattr(retrieval_module.asyncio, "gather", spy_gather)

    results = asyncio.run(
        retriever.retrieve_many(
            [
                CatalogRetrievalQuery(category="cabinets", style="modern"),
                CatalogRetrievalQuery(category="appliances", query="dishwasher"),
                CatalogRetrievalQuery(category="sinks"),
            ]
        )
    )

    assert gather_call_sizes == [3]
    assert [result.category for result in results] == ["cabinets", "appliances", "sinks"]
    assert [product.id for product in results[1].products] == ["SKU-A03", "SKU-A08"]


def test_retrieve_categories_returns_results_by_partition() -> None:
    retriever = build_retriever()

    results = asyncio.run(
        retriever.retrieve_categories(["cabinets", "appliances", "sinks"], style="modern")
    )

    assert set(results) == {"cabinets", "appliances", "sinks"}
    assert results["cabinets"].count > 0
    assert results["appliances"].count > 0
    assert results["sinks"].count == 1


def test_query_and_limit_are_applied_after_cache_lookup() -> None:
    retriever = build_retriever()

    all_appliances = asyncio.run(
        retriever.retrieve_category(CatalogRetrievalQuery(category="appliances", limit=100))
    )
    dishwashers = asyncio.run(
        retriever.retrieve_category(
            CatalogRetrievalQuery(category="appliances", query="dishwasher", limit=1)
        )
    )

    assert all_appliances.total == 8
    assert dishwashers.total == 2
    assert [product.id for product in dishwashers.products] == ["SKU-A03"]
    assert dishwashers.cache_hit is True


def test_unknown_category_is_rejected_before_catalog_scan() -> None:
    retriever = build_retriever()

    with pytest.raises(UnknownCatalogPartitionError, match="countertops"):
        asyncio.run(
            retriever.retrieve_category(CatalogRetrievalQuery(category="countertops"))
        )
