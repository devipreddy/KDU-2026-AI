from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from auto_design.catalog.service import CatalogService
from auto_design.geometry import DimensionsMM, PointMM, RoomBounds
from auto_design.planner.grammar import KitchenTopologyTemplate, RunRole, ZoneType
from auto_design.planner.zones import MacroZonePlan
from auto_design.schemas.catalog import Product
from auto_design.schemas.environment import Environment, WallAnchor
from auto_design.schemas.intent import LayoutFamilyCode


MAX_CONTINUITY_GAP_MM = 50.0
COMPACT_BASE_WIDTH_MM = 450.0
HOOD_CENTER_Z_MM = 1900.0
WALL_CABINET_CENTER_Z_MM = 1750.0

PlacementLayer = Literal["base_run", "overhead"]

FLOOR_RUN_COMPONENTS = {
    "base_cabinet",
    "dishwasher",
    "double_sink",
    "fridge",
    "prep_base_cabinet",
    "single_sink",
    "sink",
    "stove",
    "tall_cabinet",
}
OVERHEAD_COMPONENTS = {"hood", "wall_cabinet"}
BACKED_COMPONENTS = {
    "dishwasher",
    "double_sink",
    "fridge",
    "hood",
    "single_sink",
    "sink",
    "stove",
}
BASE_COMPONENTS = {"base_cabinet", "prep_base_cabinet"}


@dataclass(frozen=True)
class SequenceComponent:
    component: str
    zone_type: ZoneType
    product: Product
    optional: bool = False
    terminator: bool = False

    @property
    def width_mm(self) -> float:
        return float(self.product.width_mm)


@dataclass(frozen=True)
class PlacedItem:
    key: str
    component: str
    product_id: str
    product_type: str
    wall: WallAnchor
    run_role: RunRole
    zone_type: ZoneType
    layer: PlacementLayer
    sequence_index: int
    start_mm: float
    end_mm: float
    position_mm: PointMM
    dimensions_mm: DimensionsMM
    rotation_z_deg: float
    backed_by_product_id: str | None = None
    backing_width_mm: float | None = None
    is_base_cabinet: bool = False
    is_terminator: bool = False

    @property
    def width_mm(self) -> float:
        return self.end_mm - self.start_mm

    def to_layout_item(self) -> dict[str, object]:
        return {
            "product_id": self.product_id,
            "position_mm": self.position_mm.to_dict(),
            "dimensions_mm": self.dimensions_mm.to_dict(),
            "rotation_z_deg": self.rotation_z_deg,
            "anchor_wall": self.wall,
            "zone_type": self.zone_type,
        }

    def to_payload(self) -> dict[str, object]:
        payload = {
            "key": self.key,
            "component": self.component,
            "product_id": self.product_id,
            "product_type": self.product_type,
            "wall": self.wall,
            "run_role": self.run_role,
            "zone_type": self.zone_type,
            "layer": self.layer,
            "sequence_index": self.sequence_index,
            "start_mm": self.start_mm,
            "end_mm": self.end_mm,
            "width_mm": self.width_mm,
            "position_mm": self.position_mm.to_dict(),
            "dimensions_mm": self.dimensions_mm.to_dict(),
            "rotation_z_deg": self.rotation_z_deg,
            "backed_by_product_id": self.backed_by_product_id,
            "backing_width_mm": self.backing_width_mm,
            "is_base_cabinet": self.is_base_cabinet,
            "is_terminator": self.is_terminator,
        }
        return payload


@dataclass(frozen=True)
class PlacementRun:
    wall: WallAnchor
    run_role: RunRole
    start_mm: float
    end_mm: float
    items: tuple[PlacedItem, ...]

    @property
    def base_items(self) -> tuple[PlacedItem, ...]:
        return tuple(item for item in self.items if item.layer == "base_run")

    @property
    def continuity_gaps_mm(self) -> tuple[float, ...]:
        return tuple(
            self.base_items[index + 1].start_mm - self.base_items[index].end_mm
            for index in range(len(self.base_items) - 1)
        )

    @property
    def is_continuous(self) -> bool:
        return all(abs(gap) <= MAX_CONTINUITY_GAP_MM for gap in self.continuity_gaps_mm)

    @property
    def starts_with_base(self) -> bool:
        return bool(self.base_items and self.base_items[0].is_base_cabinet)

    @property
    def ends_with_base(self) -> bool:
        return bool(self.base_items and self.base_items[-1].is_base_cabinet)

    def to_payload(self) -> dict[str, object]:
        return {
            "wall": self.wall,
            "run_role": self.run_role,
            "start_mm": self.start_mm,
            "end_mm": self.end_mm,
            "items": [item.to_payload() for item in self.items],
            "continuity_gaps_mm": list(self.continuity_gaps_mm),
            "is_continuous": self.is_continuous,
            "starts_with_base": self.starts_with_base,
            "ends_with_base": self.ends_with_base,
        }


@dataclass(frozen=True)
class PlacementPlan:
    template_id: str
    family: LayoutFamilyCode
    runs: tuple[PlacementRun, ...]
    overhead_items: tuple[PlacedItem, ...]
    rationale: tuple[str, ...]

    @property
    def items(self) -> tuple[PlacedItem, ...]:
        return tuple(item for run in self.runs for item in run.items) + self.overhead_items

    @property
    def is_continuous(self) -> bool:
        return all(run.is_continuous for run in self.runs)

    def layout_payload(self) -> dict[str, dict[str, object]]:
        return {item.key: item.to_layout_item() for item in self.items}

    def to_payload(self) -> dict[str, object]:
        return {
            "template_id": self.template_id,
            "family": self.family,
            "runs": [run.to_payload() for run in self.runs],
            "overhead_items": [item.to_payload() for item in self.overhead_items],
            "item_count": len(self.items),
            "is_continuous": self.is_continuous,
            "rationale": list(self.rationale),
        }


def _product_dimensions(product: Product) -> DimensionsMM:
    return DimensionsMM(
        width=float(product.width_mm),
        depth=float(product.depth_mm),
        height=float(product.height_mm),
    )


def _products_matching(catalog: CatalogService, *needles: str) -> tuple[Product, ...]:
    lowered_needles = tuple(needle.casefold() for needle in needles)
    return tuple(
        product
        for product in catalog.products
        if any(needle in product.type.casefold() for needle in lowered_needles)
    )


def _choose_narrowest(products: tuple[Product, ...]) -> Product:
    if not products:
        raise ValueError("No catalog product matched placement requirement.")
    return min(products, key=lambda product: (float(product.width_mm), product.id))


def _base_candidates(catalog: CatalogService, *, min_width_mm: float = 0.0) -> tuple[Product, ...]:
    return tuple(
        product
        for product in catalog.by_category("cabinet")
        if product.type.startswith(("base_cabinet", "base_drawer"))
        and float(product.depth_mm) == 600.0
        and float(product.width_mm) >= min_width_mm
    )


def _choose_base(catalog: CatalogService, *, min_width_mm: float = 0.0) -> Product:
    candidates = _base_candidates(catalog, min_width_mm=min_width_mm)
    if candidates:
        return _choose_narrowest(candidates)
    fallback = tuple(
        product
        for product in catalog.by_category("cabinet")
        if "base" in product.type and float(product.width_mm) >= min_width_mm
    )
    return _choose_narrowest(fallback)


def _product_for_component(catalog: CatalogService, component: str) -> Product:
    if component in {"base_cabinet", "prep_base_cabinet"}:
        return _choose_base(catalog, min_width_mm=COMPACT_BASE_WIDTH_MM)
    if component == "dishwasher":
        return _choose_narrowest(_products_matching(catalog, "dishwasher"))
    if component == "double_sink":
        return _choose_narrowest(_products_matching(catalog, "sink_double"))
    if component in {"single_sink", "sink"}:
        return _choose_narrowest(_products_matching(catalog, "sink_single"))
    if component == "fridge":
        return _choose_narrowest(_products_matching(catalog, "fridge", "refrigerator"))
    if component == "hood":
        return _choose_narrowest(_products_matching(catalog, "hood"))
    if component == "stove":
        return _choose_narrowest(_products_matching(catalog, "stove", "cooktop"))
    if component == "tall_cabinet":
        return _choose_narrowest(_products_matching(catalog, "tall_cabinet"))
    if component == "wall_cabinet":
        return _choose_narrowest(_products_matching(catalog, "wall_cabinet"))
    raise ValueError(f"Unsupported placement component: {component}")


def _backing_for_component(
    catalog: CatalogService,
    component: str,
    width_mm: float,
) -> Product | None:
    if component not in BACKED_COMPONENTS:
        return None
    return _choose_base(catalog, min_width_mm=width_mm)


def _base_sequence_from_zone_plan(
    template: KitchenTopologyTemplate,
    zone_plan: MacroZonePlan,
    wall: WallAnchor,
    catalog: CatalogService,
) -> list[SequenceComponent]:
    zone_components = {
        (zone.zone_type, zone.wall): zone.components
        for zone in zone_plan.zones
    }
    sequence: list[SequenceComponent] = []
    seen: set[tuple[str, ZoneType]] = set()

    for step in template.steps:
        if step.wall != wall:
            continue
        components = zone_components.get((step.zone_type, step.wall), ())
        for component in components:
            if component == "prep_counter" or component not in FLOOR_RUN_COMPONENTS:
                continue
            key = (component, step.zone_type)
            if key in seen:
                continue
            seen.add(key)
            sequence.append(
                SequenceComponent(
                    component=component,
                    zone_type=step.zone_type,
                    product=_product_for_component(catalog, component),
                    optional=component in {"base_cabinet", "prep_base_cabinet"},
                )
            )
    return sequence


def _with_base_terminators(
    sequence: list[SequenceComponent],
    catalog: CatalogService,
) -> list[SequenceComponent]:
    terminator_product = _choose_base(catalog, min_width_mm=COMPACT_BASE_WIDTH_MM)
    result = list(sequence)
    if not result or result[0].component not in BASE_COMPONENTS:
        result.insert(
            0,
            SequenceComponent(
                component="base_cabinet",
                zone_type="storage",
                product=terminator_product,
                terminator=True,
            ),
        )
    if not result or result[-1].component not in BASE_COMPONENTS:
        result.append(
            SequenceComponent(
                component="base_cabinet",
                zone_type="storage",
                product=terminator_product,
                terminator=True,
            )
        )
    return result


def _sequence_width(sequence: list[SequenceComponent]) -> float:
    return sum(component.width_mm for component in sequence)


def _compact_to_fit(
    sequence: list[SequenceComponent],
    *,
    max_width_mm: float,
) -> list[SequenceComponent]:
    result = list(sequence)
    for optional_component in ("base_cabinet", "prep_base_cabinet"):
        while _sequence_width(result) > max_width_mm:
            removable_index = next(
                (
                    index
                    for index, component in enumerate(result)
                    if component.optional
                    and not component.terminator
                    and component.component == optional_component
                ),
                None,
            )
            if removable_index is None:
                break
            result.pop(removable_index)
    return result


def _place_base_item(
    *,
    key: str,
    sequence_index: int,
    component: SequenceComponent,
    wall: WallAnchor,
    run_role: RunRole,
    start_mm: float,
    bounds: RoomBounds,
    catalog: CatalogService,
) -> PlacedItem:
    product = component.product
    dimensions = _product_dimensions(product)
    end_mm = start_mm + dimensions.width
    wall_system = bounds.wall(wall)
    backing = _backing_for_component(catalog, component.component, dimensions.width)
    return PlacedItem(
        key=key,
        component=component.component,
        product_id=product.id,
        product_type=product.type,
        wall=wall,
        run_role=run_role,
        zone_type=component.zone_type,
        layer="base_run",
        sequence_index=sequence_index,
        start_mm=start_mm,
        end_mm=end_mm,
        position_mm=wall_system.center_for(
            offset_mm=(start_mm + end_mm) / 2.0,
            depth_mm=dimensions.depth,
            height_mm=dimensions.height,
        ),
        dimensions_mm=dimensions,
        rotation_z_deg=wall_system.rotation_z_deg,
        backed_by_product_id=backing.id if backing is not None else None,
        backing_width_mm=float(backing.width_mm) if backing is not None else None,
        is_base_cabinet=component.component in BASE_COMPONENTS,
        is_terminator=component.terminator,
    )


def _overhead_components_for_wall(
    zone_plan: MacroZonePlan,
    wall: WallAnchor,
) -> tuple[tuple[str, ZoneType], ...]:
    components: list[tuple[str, ZoneType]] = []
    for zone in zone_plan.zones:
        if zone.wall != wall:
            continue
        for component in zone.components:
            if component in OVERHEAD_COMPONENTS:
                components.append((component, zone.zone_type))
    return tuple(components)


def _anchor_for_overhead(
    component: str,
    base_items: tuple[PlacedItem, ...],
) -> PlacedItem | None:
    if component == "hood":
        return next((item for item in base_items if item.component == "stove"), None)
    if component == "wall_cabinet":
        return next((item for item in base_items if item.is_base_cabinet), None)
    return None


def _place_overhead_item(
    *,
    key: str,
    component: str,
    zone_type: ZoneType,
    anchor: PlacedItem,
    catalog: CatalogService,
    bounds: RoomBounds,
) -> PlacedItem:
    product = _product_for_component(catalog, component)
    dimensions = _product_dimensions(product)
    wall_system = bounds.wall(anchor.wall)
    center_z = HOOD_CENTER_Z_MM if component == "hood" else WALL_CABINET_CENTER_Z_MM
    center_offset = (anchor.start_mm + anchor.end_mm) / 2.0
    backing = _backing_for_component(catalog, component, dimensions.width)
    return PlacedItem(
        key=key,
        component=component,
        product_id=product.id,
        product_type=product.type,
        wall=anchor.wall,
        run_role=anchor.run_role,
        zone_type=zone_type,
        layer="overhead",
        sequence_index=anchor.sequence_index,
        start_mm=center_offset - (dimensions.width / 2.0),
        end_mm=center_offset + (dimensions.width / 2.0),
        position_mm=wall_system.center_for(
            offset_mm=center_offset,
            depth_mm=dimensions.depth,
            height_mm=dimensions.height,
            z_center_mm=center_z,
        ),
        dimensions_mm=dimensions,
        rotation_z_deg=wall_system.rotation_z_deg,
        backed_by_product_id=backing.id if backing is not None else None,
        backing_width_mm=float(backing.width_mm) if backing is not None else None,
        is_base_cabinet=False,
        is_terminator=False,
    )


def generate_placement_plan(
    environment: Environment,
    template: KitchenTopologyTemplate,
    zone_plan: MacroZonePlan,
    catalog: CatalogService,
) -> PlacementPlan:
    bounds = RoomBounds.from_environment(environment)
    placement_runs: list[PlacementRun] = []
    overhead_items: list[PlacedItem] = []
    item_counts: dict[str, int] = {}

    for run in template.runs:
        sequence = _base_sequence_from_zone_plan(template, zone_plan, run.wall, catalog)
        sequence = _with_base_terminators(sequence, catalog)
        sequence = _compact_to_fit(sequence, max_width_mm=run.length_mm)

        cursor = run.start_mm
        placed_items: list[PlacedItem] = []
        for sequence_index, component in enumerate(sequence, start=1):
            item_counts[component.component] = item_counts.get(component.component, 0) + 1
            key = f"{component.component}_{run.wall}_{item_counts[component.component]}"
            item = _place_base_item(
                key=key,
                sequence_index=sequence_index,
                component=component,
                wall=run.wall,
                run_role=run.role,
                start_mm=cursor,
                bounds=bounds,
                catalog=catalog,
            )
            placed_items.append(item)
            cursor = item.end_mm

        base_items = tuple(placed_items)
        for component, zone_type in _overhead_components_for_wall(zone_plan, run.wall):
            anchor = _anchor_for_overhead(component, base_items)
            if anchor is None:
                continue
            item_counts[component] = item_counts.get(component, 0) + 1
            overhead_items.append(
                _place_overhead_item(
                    key=f"{component}_{run.wall}_{item_counts[component]}",
                    component=component,
                    zone_type=zone_type,
                    anchor=anchor,
                    catalog=catalog,
                    bounds=bounds,
                )
            )

        placement_runs.append(
            PlacementRun(
                wall=run.wall,
                run_role=run.role,
                start_mm=run.start_mm,
                end_mm=cursor,
                items=tuple(placed_items),
            )
        )

    return PlacementPlan(
        template_id=template.id,
        family=template.family,
        runs=tuple(placement_runs),
        overhead_items=tuple(overhead_items),
        rationale=(
            "Items are placed as snapped wall-run sequences, never as free-floating boxes.",
            "Each base run starts and ends with a base cabinet terminator.",
            "Adjacent base-run spans are generated from the previous item end offset.",
            "Appliances and sinks carry equal-or-larger base backing metadata.",
        ),
    )
