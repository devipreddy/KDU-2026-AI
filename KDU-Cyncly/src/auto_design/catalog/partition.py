from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from auto_design.catalog.service import CatalogService
from auto_design.schemas.catalog import Product


CatalogPartition = Literal["cabinets", "appliances", "sinks", "fixtures", "doors", "windows"]

PARTITION_FILENAMES: Mapping[CatalogPartition, str] = {
    "cabinets": "cabinets.json",
    "appliances": "appliances.json",
    "sinks": "sinks.json",
    "fixtures": "fixtures.json",
    "doors": "doors.json",
    "windows": "windows.json",
}


def partition_name_for(product: Product) -> CatalogPartition:
    if product.category == "cabinet":
        return "cabinets"
    if product.category == "appliance":
        return "appliances"
    if product.category == "sink" or product.type.startswith("sink_"):
        return "sinks"
    if product.category == "fixture":
        return "fixtures"
    if product.category == "door":
        return "doors"
    if product.category == "window":
        return "windows"
    raise ValueError(f"Unsupported catalog category for {product.id}: {product.category}")


def derive_partitions(products: Iterable[Product]) -> dict[CatalogPartition, tuple[Product, ...]]:
    partitions: dict[CatalogPartition, list[Product]] = {
        name: [] for name in PARTITION_FILENAMES
    }
    for product in products:
        partitions[partition_name_for(product)].append(product)
    return {name: tuple(items) for name, items in partitions.items()}


def serialize_products(products: Iterable[Product]) -> list[dict[str, object]]:
    return [product.model_dump(mode="json") for product in products]


def write_partitions(source_catalog: str | Path, output_dir: str | Path) -> dict[CatalogPartition, Path]:
    catalog = CatalogService.load(source_catalog)
    partitions = derive_partitions(catalog.products)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[CatalogPartition, Path] = {}
    for name, filename in PARTITION_FILENAMES.items():
        path = out_dir / filename
        payload = serialize_products(partitions[name])
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written[name] = path
    return written


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive category-specific catalog partitions from catalog.json."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("catalog.json"),
        help="Source-of-truth catalog JSON path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("catalog"),
        help="Directory for derived partition JSON files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    written = write_partitions(args.catalog, args.out_dir)
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
