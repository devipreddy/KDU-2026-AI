from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Self

from auto_design.schemas.catalog import Product, ProductCatalog, ProductCategory


class CatalogError(ValueError):
    """Base class for catalog validation failures."""


class DuplicateSkuError(CatalogError):
    """Raised when the source catalog defines the same SKU more than once."""


class UnknownSkuError(CatalogError):
    """Raised when planner output references a SKU outside the catalog."""


def _normalize_color(color: str) -> str:
    return color.upper()


@dataclass(frozen=True)
class CatalogService:
    """Read-only catalog service backed by the source-of-truth catalog JSON."""

    source_path: Path | None
    products: tuple[Product, ...]
    by_sku: dict[str, Product]
    by_category_index: dict[str, tuple[Product, ...]]
    by_type_index: dict[str, tuple[Product, ...]]
    by_color_index: dict[str, tuple[Product, ...]]

    @classmethod
    def load(cls, path: str | Path) -> Self:
        catalog_path = Path(path)
        with catalog_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        catalog = ProductCatalog.model_validate(payload)
        return cls.from_products(catalog.root, source_path=catalog_path)

    @classmethod
    def from_products(cls, products: Iterable[Product], source_path: str | Path | None = None) -> Self:
        product_list = tuple(products)
        by_sku: dict[str, Product] = {}
        category_lists: dict[str, list[Product]] = defaultdict(list)
        type_lists: dict[str, list[Product]] = defaultdict(list)
        color_lists: dict[str, list[Product]] = defaultdict(list)

        for product in product_list:
            if product.id in by_sku:
                raise DuplicateSkuError(f"Duplicate catalog SKU: {product.id}")
            by_sku[product.id] = product
            category_lists[product.category].append(product)
            type_lists[product.type].append(product)
            color_lists[_normalize_color(product.color)].append(product)

        resolved_source = Path(source_path) if source_path is not None else None
        return cls(
            source_path=resolved_source,
            products=product_list,
            by_sku=by_sku,
            by_category_index={key: tuple(value) for key, value in category_lists.items()},
            by_type_index={key: tuple(value) for key, value in type_lists.items()},
            by_color_index={key: tuple(value) for key, value in color_lists.items()},
        )

    def __len__(self) -> int:
        return len(self.products)

    def has_sku(self, sku: str) -> bool:
        return sku in self.by_sku

    def get(self, sku: str) -> Product:
        try:
            return self.by_sku[sku]
        except KeyError as exc:
            raise UnknownSkuError(f"Unknown catalog SKU: {sku}") from exc

    def require_many(self, skus: Iterable[str]) -> dict[str, Product]:
        requested = tuple(dict.fromkeys(skus))
        missing = [sku for sku in requested if sku not in self.by_sku]
        if missing:
            joined = ", ".join(missing)
            raise UnknownSkuError(f"Unknown catalog SKU(s): {joined}")
        return {sku: self.by_sku[sku] for sku in requested}

    def validate_skus(self, skus: Iterable[str]) -> None:
        self.require_many(skus)

    def by_category(self, category: ProductCategory | str) -> tuple[Product, ...]:
        return self.by_category_index.get(str(category), ())

    def by_type(self, product_type: str) -> tuple[Product, ...]:
        return self.by_type_index.get(product_type, ())

    def by_color(self, color: str) -> tuple[Product, ...]:
        return self.by_color_index.get(_normalize_color(color), ())
