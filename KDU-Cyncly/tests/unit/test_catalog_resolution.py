from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from layout import LayoutVisualizer  # noqa: E402


def test_known_skus_load_from_catalog_json() -> None:
    visualizer = LayoutVisualizer(catalog=ROOT / "catalog.json", strict=True)

    assert "SKU-C01" in visualizer._catalog
    assert visualizer._catalog["SKU-C01"]["type"] == "base_cabinet_600"
    assert visualizer._catalog["SKU-C01"]["width_mm"] == 600
    assert visualizer._catalog["SKU-C01"]["depth_mm"] == 600
    assert visualizer._catalog["SKU-C01"]["height_mm"] == 900


def test_catalog_dimensions_win_over_json_dimensions_for_known_sku() -> None:
    visualizer = LayoutVisualizer(catalog=ROOT / "catalog.json", strict=True)

    dims, label, item_type = visualizer._resolve_dims_and_label(
        "base_cabinet_1",
        {
            "product_id": "SKU-C01",
            "dimensions_mm": {"width": 300, "depth": 300, "height": 500},
            "position_mm": {"x": 1000, "y": 2700, "z": 450},
            "rotation_z_deg": 180,
        },
    )

    assert dims == {"width": 600.0, "depth": 600.0, "height": 900.0}
    assert label == "SKU-C01"
    assert item_type == "base_cabinet_600"
