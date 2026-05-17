from __future__ import annotations

import json
from pathlib import Path

from auto_design.catalog import CatalogService
from auto_design.catalog.partition import (
    PARTITION_FILENAMES,
    derive_partitions,
    partition_name_for,
    serialize_products,
    write_partitions,
)


ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "catalog.json"
PARTITION_DIR = ROOT / "catalog"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_partition_routing_splits_sinks_from_other_fixtures() -> None:
    catalog = CatalogService.load(CATALOG_PATH)

    assert partition_name_for(catalog.get("SKU-S01")) == "sinks"
    assert partition_name_for(catalog.get("SKU-S02")) == "sinks"
    assert partition_name_for(catalog.get("SKU-F01")) == "fixtures"


def test_derived_partitions_have_expected_counts() -> None:
    catalog = CatalogService.load(CATALOG_PATH)
    partitions = derive_partitions(catalog.products)

    assert {name: len(products) for name, products in partitions.items()} == {
        "cabinets": 17,
        "appliances": 8,
        "sinks": 2,
        "fixtures": 1,
        "doors": 0,
        "windows": 0,
    }


def test_checked_in_partition_files_are_synchronized_with_source_catalog() -> None:
    catalog = CatalogService.load(CATALOG_PATH)
    expected = derive_partitions(catalog.products)
    seen_ids: list[str] = []

    for name, filename in PARTITION_FILENAMES.items():
        path = PARTITION_DIR / filename
        expected_payload = serialize_products(expected[name])
        actual_payload = load_json(path)

        assert path.is_file()
        assert actual_payload == expected_payload
        assert isinstance(actual_payload, list)
        seen_ids.extend(item["id"] for item in actual_payload)

    root_ids = [product.id for product in catalog.products]
    assert set(seen_ids) == set(root_ids)
    assert len(seen_ids) == len(set(seen_ids))


def test_partition_writer_generates_the_same_payloads_in_a_target_directory(tmp_path: Path) -> None:
    written = write_partitions(CATALOG_PATH, tmp_path)

    assert set(written) == set(PARTITION_FILENAMES)
    for name, filename in PARTITION_FILENAMES.items():
        checked_in_payload = load_json(PARTITION_DIR / filename)
        generated_payload = load_json(tmp_path / filename)

        assert written[name] == tmp_path / filename
        assert generated_payload == checked_in_payload
