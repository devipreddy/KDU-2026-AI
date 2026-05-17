from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_design.catalog import CatalogService, DuplicateSkuError, UnknownSkuError
from auto_design.schemas.catalog import ProductCatalog


ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "catalog.json"


def test_catalog_service_loads_source_catalog_by_sku() -> None:
    catalog = CatalogService.load(CATALOG_PATH)

    product = catalog.get("SKU-C01")

    assert len(catalog) == 28
    assert product.id == "SKU-C01"
    assert product.type == "base_cabinet_600"
    assert product.width_mm == 600
    assert catalog.source_path == CATALOG_PATH


def test_catalog_service_indexes_by_category_type_and_color() -> None:
    catalog = CatalogService.load(CATALOG_PATH)

    cabinet_ids = {product.id for product in catalog.by_category("cabinet")}
    type_ids = {product.id for product in catalog.by_type("base_cabinet_600")}
    navy_ids = {product.id for product in catalog.by_color("#1f3a5f")}

    assert len(cabinet_ids) == 17
    assert "SKU-C11" in cabinet_ids
    assert type_ids == {"SKU-C01"}
    assert navy_ids == {"SKU-C11"}


def test_catalog_service_rejects_invented_single_sku() -> None:
    catalog = CatalogService.load(CATALOG_PATH)

    with pytest.raises(UnknownSkuError, match="SKU-DOES-NOT-EXIST"):
        catalog.get("SKU-DOES-NOT-EXIST")


def test_catalog_service_rejects_invented_skus_in_batches() -> None:
    catalog = CatalogService.load(CATALOG_PATH)

    with pytest.raises(UnknownSkuError, match="SKU-FAKE"):
        catalog.require_many(["SKU-C01", "SKU-FAKE", "SKU-A01"])


def test_catalog_service_validates_known_skus_without_duplicates_in_result() -> None:
    catalog = CatalogService.load(CATALOG_PATH)

    products = catalog.require_many(["SKU-C01", "SKU-A01", "SKU-C01"])

    assert list(products) == ["SKU-C01", "SKU-A01"]


def test_catalog_service_rejects_duplicate_source_skus() -> None:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    parsed = ProductCatalog.model_validate(payload)
    duplicate_products = [*parsed.root, parsed.root[0]]

    with pytest.raises(DuplicateSkuError, match=parsed.root[0].id):
        CatalogService.from_products(duplicate_products)
