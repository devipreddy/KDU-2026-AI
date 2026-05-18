from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from auto_design.schemas import DesignInput, LayoutItem, LayoutResponse, ProductCatalog


ROOT = Path(__file__).resolve().parents[2]


def load_json(name: str) -> object:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("sample_name", ["input1.json", "input2.json", "input3.json"])
def test_design_inputs_validate_against_contract(sample_name: str) -> None:
    payload = load_json(sample_name)

    parsed = DesignInput.model_validate(payload)

    assert parsed.environment.floor.points
    assert parsed.environment.wall
    assert parsed.preferences.prompt


def test_preferences_accept_catalog_id_alias() -> None:
    payload = load_json("input1.json")
    assert isinstance(payload, dict)
    assert isinstance(payload["preferences"], dict)
    payload["preferences"].pop("catalog", None)
    payload["preferences"]["catalogId"] = "starter-catalog"

    parsed = DesignInput.model_validate(payload)

    assert parsed.preferences.catalog_id == "starter-catalog"


def test_catalog_validates_products_and_dimensions() -> None:
    catalog = ProductCatalog.model_validate(load_json("catalog.json"))

    sku_by_id = {product.id: product for product in catalog.root}

    assert len(catalog.root) == 28
    assert sku_by_id["SKU-C01"].category == "cabinet"
    assert sku_by_id["SKU-C01"].width_mm == 600


def test_output_envelope_validates_renderer_contract() -> None:
    response = LayoutResponse.model_validate(load_json("output.json"))

    assert str(response.request_id) == "00000000-0000-0000-0000-000000000001"
    assert len(response.layouts) == 5
    assert response.layouts[0].layout["sink_1"].product_id == "SKU-S01"


def test_schema_contracts_reject_unexpected_fields() -> None:
    payload = load_json("input1.json")
    assert isinstance(payload, dict)
    assert isinstance(payload["preferences"], dict)
    payload["preferences"]["unexpected"] = "not allowed"

    with pytest.raises(ValidationError):
        DesignInput.model_validate(payload)


def test_layout_item_requires_structure_or_product_identity() -> None:
    with pytest.raises(ValidationError):
        LayoutItem.model_validate(
            {
                "position_mm": {"x": 0, "y": 0, "z": 0},
                "rotation_z_deg": 0,
            }
        )
