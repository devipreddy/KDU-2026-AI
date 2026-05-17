from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from auto_design.catalog.partition import CatalogPartition, PARTITION_FILENAMES, partition_name_for
from auto_design.catalog.service import CatalogService
from auto_design.schemas.catalog import Product


class UnknownCatalogPartitionError(ValueError):
    """Raised when retrieval is requested for a partition the catalog does not expose."""


def normalize_partition(category: CatalogPartition | str) -> CatalogPartition:
    normalized = category.casefold()
    if normalized not in PARTITION_FILENAMES:
        raise UnknownCatalogPartitionError(f"Unknown catalog partition: {category}")
    return cast(CatalogPartition, normalized)


def normalize_color(color: str | None) -> str | None:
    return color.upper() if color else None


def normalize_style(style: str | None) -> str | None:
    return style.casefold() if style else None


def products_for_partition(
    products: Iterable[Product],
    category: CatalogPartition | str,
) -> tuple[Product, ...]:
    partition = normalize_partition(category)
    return tuple(product for product in products if partition_name_for(product) == partition)


def _matches_query(product: Product, query: str) -> bool:
    if not query:
        return True
    needle = query.casefold()
    searchable = [
        product.id,
        product.type,
        product.category,
        product.color,
        product.price_tier,
        *product.style_tags,
    ]
    return any(needle in value.casefold() for value in searchable)


def filter_base_candidates(
    products: Iterable[Product],
    *,
    category: CatalogPartition | str,
    style: str | None = None,
    color: str | None = None,
) -> tuple[Product, ...]:
    partition_products = products_for_partition(products, category)
    normalized_style = normalize_style(style)
    normalized_color = normalize_color(color)

    matches: list[Product] = []
    for product in partition_products:
        product_styles = {tag.casefold() for tag in product.style_tags}
        if normalized_style and normalized_style not in product_styles:
            continue
        if normalized_color and product.color.upper() != normalized_color:
            continue
        matches.append(product)
    return tuple(matches)


def filter_products(
    products: Iterable[Product],
    *,
    query: str = "",
    price_tier: str | None = None,
    product_type: str | None = None,
    limit: int = 20,
) -> tuple[int, tuple[Product, ...]]:
    normalized_price = price_tier.casefold() if price_tier else None
    normalized_type = product_type.casefold() if product_type else None

    matches: list[Product] = []
    for product in products:
        if normalized_price and product.price_tier.casefold() != normalized_price:
            continue
        if normalized_type and product.type.casefold() != normalized_type:
            continue
        if not _matches_query(product, query):
            continue
        matches.append(product)

    safe_limit = max(0, min(limit, 100))
    return len(matches), tuple(matches[:safe_limit])


@dataclass(frozen=True)
class RetrievalCacheKey:
    category: CatalogPartition
    style: str | None
    color: str | None

    @classmethod
    def from_query(cls, query: CatalogRetrievalQuery) -> RetrievalCacheKey:
        return cls(
            category=normalize_partition(query.category),
            style=normalize_style(query.style),
            color=normalize_color(query.color),
        )


@dataclass(frozen=True)
class CatalogRetrievalQuery:
    category: CatalogPartition | str
    style: str | None = None
    color: str | None = None
    query: str = ""
    price_tier: str | None = None
    product_type: str | None = None
    limit: int = 20

    @property
    def cache_key(self) -> RetrievalCacheKey:
        return RetrievalCacheKey.from_query(self)


@dataclass(frozen=True)
class CatalogRetrievalResult:
    category: CatalogPartition
    products: tuple[Product, ...]
    total: int
    cache_hit: bool
    cache_key: RetrievalCacheKey

    @property
    def count(self) -> int:
        return len(self.products)

    def to_payload(self) -> dict[str, object]:
        return {
            "category": self.category,
            "count": self.count,
            "total": self.total,
            "cache_hit": self.cache_hit,
            "products": [product.model_dump(mode="json") for product in self.products],
        }


class AsyncCatalogRetriever:
    """Async catalog retrieval with a semantic in-memory candidate cache."""

    def __init__(self, catalog: CatalogService) -> None:
        self.catalog = catalog
        self._cache: dict[RetrievalCacheKey, tuple[Product, ...]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def cache_keys(self) -> tuple[RetrievalCacheKey, ...]:
        return tuple(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    async def _load_base_candidates(self, key: RetrievalCacheKey) -> tuple[Product, ...]:
        await asyncio.sleep(0)
        return filter_base_candidates(
            self.catalog.products,
            category=key.category,
            style=key.style,
            color=key.color,
        )

    async def retrieve_category(self, query: CatalogRetrievalQuery) -> CatalogRetrievalResult:
        key = query.cache_key
        if key in self._cache:
            self.cache_hits += 1
            cache_hit = True
            candidates = self._cache[key]
        else:
            self.cache_misses += 1
            cache_hit = False
            candidates = await self._load_base_candidates(key)
            self._cache[key] = candidates

        total, products = filter_products(
            candidates,
            query=query.query,
            price_tier=query.price_tier,
            product_type=query.product_type,
            limit=query.limit,
        )
        return CatalogRetrievalResult(
            category=key.category,
            products=products,
            total=total,
            cache_hit=cache_hit,
            cache_key=key,
        )

    async def retrieve_many(
        self,
        queries: Iterable[CatalogRetrievalQuery],
    ) -> tuple[CatalogRetrievalResult, ...]:
        query_batch = tuple(queries)
        if not query_batch:
            return ()
        results = await asyncio.gather(
            *(self.retrieve_category(query) for query in query_batch)
        )
        return tuple(results)

    async def retrieve_categories(
        self,
        categories: Iterable[CatalogPartition | str],
        *,
        style: str | None = None,
        color: str | None = None,
        limit: int = 20,
    ) -> dict[CatalogPartition, CatalogRetrievalResult]:
        queries = [
            CatalogRetrievalQuery(
                category=category,
                style=style,
                color=color,
                limit=limit,
            )
            for category in categories
        ]
        results = await self.retrieve_many(queries)
        return {result.category: result for result in results}
