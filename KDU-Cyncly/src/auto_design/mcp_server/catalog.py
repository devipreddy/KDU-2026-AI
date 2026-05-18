from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from auto_design.catalog.retrieval import filter_base_candidates, filter_products
from auto_design.catalog.service import CatalogService
from auto_design.schemas.catalog import Product


SearchPartition = Literal["cabinets", "appliances", "sinks", "fixtures"]


def product_payload(product: Product) -> dict[str, Any]:
    return product.model_dump(mode="json")


def _catalog_source(catalog: CatalogService) -> str | None:
    return str(catalog.source_path) if catalog.source_path is not None else None


def search_partition(
    catalog: CatalogService,
    partition: SearchPartition,
    *,
    query: str = "",
    color: str | None = None,
    style: str | None = None,
    price_tier: str | None = None,
    product_type: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    candidates = filter_base_candidates(
        catalog.products,
        category=partition,
        style=style,
        color=color,
    )
    total, limited = filter_products(
        candidates,
        query=query,
        price_tier=price_tier,
        product_type=product_type,
        limit=limit,
    )
    return {
        "source": _catalog_source(catalog),
        "partition": partition,
        "count": len(limited),
        "total": total,
        "products": [product_payload(product) for product in limited],
    }


def lookup_sku(catalog: CatalogService, sku: str) -> dict[str, Any]:
    return {
        "source": _catalog_source(catalog),
        "product": product_payload(catalog.get(sku)),
    }


def create_catalog_server(catalog_path: str | Path = "catalog.json") -> FastMCP:
    catalog = CatalogService.load(catalog_path)
    server = FastMCP(
        name="Cyncly Catalog MCP",
        instructions=(
            "Search and retrieve products from the kitchen catalog. "
            "All returned products are validated against catalog.json."
        ),
    )

    @server.tool(
        name="catalog.cabinets.search",
        description="Search catalog-approved cabinet SKUs.",
        structured_output=True,
    )
    def search_cabinets(
        query: str = "",
        color: str | None = None,
        style: str | None = None,
        price_tier: str | None = None,
        product_type: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        return search_partition(
            catalog,
            "cabinets",
            query=query,
            color=color,
            style=style,
            price_tier=price_tier,
            product_type=product_type,
            limit=limit,
        )

    @server.tool(
        name="catalog.appliances.search",
        description="Search catalog-approved appliance SKUs.",
        structured_output=True,
    )
    def search_appliances(
        query: str = "",
        color: str | None = None,
        style: str | None = None,
        price_tier: str | None = None,
        product_type: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        return search_partition(
            catalog,
            "appliances",
            query=query,
            color=color,
            style=style,
            price_tier=price_tier,
            product_type=product_type,
            limit=limit,
        )

    @server.tool(
        name="catalog.sinks.search",
        description="Search catalog-approved sink SKUs.",
        structured_output=True,
    )
    def search_sinks(
        query: str = "",
        color: str | None = None,
        style: str | None = None,
        price_tier: str | None = None,
        product_type: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        return search_partition(
            catalog,
            "sinks",
            query=query,
            color=color,
            style=style,
            price_tier=price_tier,
            product_type=product_type,
            limit=limit,
        )

    @server.tool(
        name="catalog.fixtures.search",
        description="Search catalog-approved non-sink fixture SKUs.",
        structured_output=True,
    )
    def search_fixtures(
        query: str = "",
        color: str | None = None,
        style: str | None = None,
        price_tier: str | None = None,
        product_type: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        return search_partition(
            catalog,
            "fixtures",
            query=query,
            color=color,
            style=style,
            price_tier=price_tier,
            product_type=product_type,
            limit=limit,
        )

    @server.tool(
        name="catalog.sku.lookup",
        description="Return one catalog product by exact SKU, rejecting invented SKUs.",
        structured_output=True,
    )
    def lookup_catalog_sku(sku: str) -> dict[str, Any]:
        return lookup_sku(catalog, sku)

    return server


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the kitchen catalog MCP server.")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("catalog.json"),
        help="Source-of-truth catalog JSON path.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
        help="MCP transport to run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    server = create_catalog_server(args.catalog)
    server.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
