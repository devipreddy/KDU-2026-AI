from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from auto_design.mcp_server.catalog import create_catalog_server


ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "catalog.json"


def run_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async def call() -> dict[str, Any]:
        server = create_catalog_server(CATALOG_PATH)
        result = await server.call_tool(name, arguments)
        assert isinstance(result, tuple)
        return result[1]

    return asyncio.run(call())


def test_mcp_server_registers_catalog_tools() -> None:
    async def list_tool_names() -> set[str]:
        server = create_catalog_server(CATALOG_PATH)
        return {tool.name for tool in await server.list_tools()}

    assert asyncio.run(list_tool_names()) == {
        "catalog.cabinets.search",
        "catalog.appliances.search",
        "catalog.sinks.search",
        "catalog.fixtures.search",
        "catalog.sku.lookup",
    }


def test_cabinet_search_filters_by_catalog_color() -> None:
    result = run_tool("catalog.cabinets.search", {"color": "#1f3a5f"})

    assert result["partition"] == "cabinets"
    assert result["count"] == 1
    assert result["total"] == 1
    assert result["products"][0]["id"] == "SKU-C11"


def test_appliance_search_filters_by_query_text() -> None:
    result = run_tool("catalog.appliances.search", {"query": "dishwasher"})

    assert result["partition"] == "appliances"
    assert {product["id"] for product in result["products"]} == {"SKU-A03", "SKU-A08"}


def test_sink_and_fixture_tools_keep_fixture_categories_partitioned() -> None:
    sinks = run_tool("catalog.sinks.search", {})
    fixtures = run_tool("catalog.fixtures.search", {})

    assert {product["id"] for product in sinks["products"]} == {"SKU-S01", "SKU-S02"}
    assert {product["id"] for product in fixtures["products"]} == {"SKU-F01"}


def test_sku_lookup_returns_exact_catalog_product() -> None:
    result = run_tool("catalog.sku.lookup", {"sku": "SKU-A01"})

    assert result["product"]["id"] == "SKU-A01"
    assert result["product"]["type"] == "stove_60"


def test_sku_lookup_rejects_invented_products() -> None:
    async def call_unknown_sku() -> None:
        server = create_catalog_server(CATALOG_PATH)
        await server.call_tool("catalog.sku.lookup", {"sku": "SKU-FAKE"})

    with pytest.raises(ToolError, match="SKU-FAKE"):
        asyncio.run(call_unknown_sku())
