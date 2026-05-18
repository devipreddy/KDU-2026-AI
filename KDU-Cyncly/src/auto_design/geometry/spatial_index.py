from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TypeAlias

from auto_design.geometry.primitives import AABB


CellCoord: TypeAlias = tuple[int, int, int]
IndexedPayload: TypeAlias = object

DEFAULT_SPATIAL_CELL_SIZE_MM = 600.0
BOUNDARY_EPSILON_MM = 1e-6


@dataclass(frozen=True)
class IndexedAABB:
    item_id: str
    bounds: AABB
    payload: IndexedPayload | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "bounds": {
                "min_x": self.bounds.min_x,
                "min_y": self.bounds.min_y,
                "min_z": self.bounds.min_z,
                "max_x": self.bounds.max_x,
                "max_y": self.bounds.max_y,
                "max_z": self.bounds.max_z,
            },
        }


@dataclass(frozen=True)
class CollisionPair:
    first_id: str
    second_id: str
    first: IndexedAABB
    second: IndexedAABB

    def to_payload(self) -> dict[str, object]:
        return {
            "first_id": self.first_id,
            "second_id": self.second_id,
            "first": self.first.to_payload(),
            "second": self.second.to_payload(),
        }


def _cell_index(value: float, cell_size_mm: float) -> int:
    return math.floor(value / cell_size_mm)


def cells_for_aabb(
    bounds: AABB,
    *,
    cell_size_mm: float = DEFAULT_SPATIAL_CELL_SIZE_MM,
) -> tuple[CellCoord, ...]:
    if cell_size_mm <= 0:
        raise ValueError("cell_size_mm must be positive.")
    if (
        bounds.max_x < bounds.min_x
        or bounds.max_y < bounds.min_y
        or bounds.max_z < bounds.min_z
    ):
        raise ValueError("AABB max bounds must be greater than or equal to min bounds.")

    max_x = bounds.max_x - BOUNDARY_EPSILON_MM
    max_y = bounds.max_y - BOUNDARY_EPSILON_MM
    max_z = bounds.max_z - BOUNDARY_EPSILON_MM
    if max_x < bounds.min_x:
        max_x = bounds.min_x
    if max_y < bounds.min_y:
        max_y = bounds.min_y
    if max_z < bounds.min_z:
        max_z = bounds.min_z

    x_range = range(
        _cell_index(bounds.min_x, cell_size_mm),
        _cell_index(max_x, cell_size_mm) + 1,
    )
    y_range = range(
        _cell_index(bounds.min_y, cell_size_mm),
        _cell_index(max_y, cell_size_mm) + 1,
    )
    z_range = range(
        _cell_index(bounds.min_z, cell_size_mm),
        _cell_index(max_z, cell_size_mm) + 1,
    )
    return tuple((x, y, z) for x in x_range for y in y_range for z in z_range)


class SpatialHashIndex:
    """Grid occupancy index for fast broad-phase spatial queries."""

    def __init__(self, *, cell_size_mm: float = DEFAULT_SPATIAL_CELL_SIZE_MM) -> None:
        if cell_size_mm <= 0:
            raise ValueError("cell_size_mm must be positive.")
        self.cell_size_mm = float(cell_size_mm)
        self._entries: dict[str, IndexedAABB] = {}
        self._entry_cells: dict[str, tuple[CellCoord, ...]] = {}
        self._cells: dict[CellCoord, set[str]] = defaultdict(set)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def occupied_cell_count(self) -> int:
        return len(self._cells)

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(self._entries)

    @property
    def entries(self) -> tuple[IndexedAABB, ...]:
        return tuple(self._entries.values())

    @classmethod
    def from_aabbs(
        cls,
        entries: Iterable[tuple[str, AABB] | tuple[str, AABB, IndexedPayload]],
        *,
        cell_size_mm: float = DEFAULT_SPATIAL_CELL_SIZE_MM,
    ) -> SpatialHashIndex:
        index = cls(cell_size_mm=cell_size_mm)
        for entry in entries:
            if len(entry) == 2:
                item_id, bounds = entry
                payload = None
            else:
                item_id, bounds, payload = entry
            index.insert(str(item_id), bounds, payload=payload)
        return index

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._entries

    def __iter__(self) -> Iterator[IndexedAABB]:
        return iter(self._entries.values())

    def get(self, item_id: str) -> IndexedAABB:
        return self._entries[item_id]

    def cells_for(self, bounds: AABB) -> tuple[CellCoord, ...]:
        return cells_for_aabb(bounds, cell_size_mm=self.cell_size_mm)

    def occupied_cells_for(self, item_id: str) -> tuple[CellCoord, ...]:
        return self._entry_cells[item_id]

    def ids_in_cell(self, cell: CellCoord) -> tuple[str, ...]:
        return tuple(sorted(self._cells.get(cell, ())))

    def insert(
        self,
        item_id: str,
        bounds: AABB,
        *,
        payload: IndexedPayload | None = None,
    ) -> None:
        if item_id in self._entries:
            self.remove(item_id)
        cells = self.cells_for(bounds)
        self._entries[item_id] = IndexedAABB(item_id=item_id, bounds=bounds, payload=payload)
        self._entry_cells[item_id] = cells
        for cell in cells:
            self._cells[cell].add(item_id)

    def remove(self, item_id: str) -> None:
        cells = self._entry_cells.pop(item_id)
        self._entries.pop(item_id)
        for cell in cells:
            members = self._cells[cell]
            members.discard(item_id)
            if not members:
                self._cells.pop(cell)

    def clear(self) -> None:
        self._entries.clear()
        self._entry_cells.clear()
        self._cells.clear()

    def query_nearby(
        self,
        bounds: AABB,
        *,
        margin_mm: float = 0.0,
        exclude_item_id: str | None = None,
    ) -> tuple[IndexedAABB, ...]:
        query_bounds = bounds.expanded(margin_mm) if margin_mm else bounds
        candidate_ids: set[str] = set()
        for cell in self.cells_for(query_bounds):
            candidate_ids.update(self._cells.get(cell, ()))
        if exclude_item_id is not None:
            candidate_ids.discard(exclude_item_id)
        return tuple(self._entries[item_id] for item_id in sorted(candidate_ids))

    def query_intersections(
        self,
        bounds: AABB,
        *,
        include_touching: bool = False,
        margin_mm: float = 0.0,
        exclude_item_id: str | None = None,
    ) -> tuple[IndexedAABB, ...]:
        actual_bounds = bounds.expanded(margin_mm) if margin_mm else bounds
        candidate_margin = margin_mm
        if include_touching:
            candidate_margin = max(candidate_margin, BOUNDARY_EPSILON_MM)
        candidate_bounds = (
            bounds.expanded(candidate_margin) if candidate_margin else bounds
        )
        return tuple(
            entry
            for entry in self.query_nearby(
                candidate_bounds,
                exclude_item_id=exclude_item_id,
            )
            if entry.bounds.intersects(actual_bounds, include_touching=include_touching)
        )

    def nearby_for_item(
        self,
        item_id: str,
        *,
        margin_mm: float = 0.0,
    ) -> tuple[IndexedAABB, ...]:
        entry = self.get(item_id)
        return self.query_nearby(
            entry.bounds,
            margin_mm=margin_mm,
            exclude_item_id=item_id,
        )

    def collisions_for_item(
        self,
        item_id: str,
        *,
        include_touching: bool = False,
        margin_mm: float = 0.0,
    ) -> tuple[IndexedAABB, ...]:
        entry = self.get(item_id)
        return self.query_intersections(
            entry.bounds,
            include_touching=include_touching,
            margin_mm=margin_mm,
            exclude_item_id=item_id,
        )

    def candidate_pairs(self) -> tuple[tuple[str, str], ...]:
        pairs: set[tuple[str, str]] = set()
        for item_ids in self._cells.values():
            ordered = sorted(item_ids)
            for first_index, first_id in enumerate(ordered):
                for second_id in ordered[first_index + 1 :]:
                    pairs.add((first_id, second_id))
        return tuple(sorted(pairs))

    def collision_pairs(self, *, include_touching: bool = False) -> tuple[CollisionPair, ...]:
        if include_touching:
            touching_pairs: dict[tuple[str, str], CollisionPair] = {}
            for first in self.entries:
                for second in self.collisions_for_item(
                    first.item_id,
                    include_touching=True,
                ):
                    first_id, second_id = sorted((first.item_id, second.item_id))
                    pair_key = (first_id, second_id)
                    if pair_key in touching_pairs:
                        continue
                    touching_pairs[pair_key] = CollisionPair(
                        first_id=first_id,
                        second_id=second_id,
                        first=self._entries[first_id],
                        second=self._entries[second_id],
                    )
            return tuple(touching_pairs[key] for key in sorted(touching_pairs))

        pairs: list[CollisionPair] = []
        for first_id, second_id in self.candidate_pairs():
            first = self._entries[first_id]
            second = self._entries[second_id]
            if not first.bounds.intersects(second.bounds):
                continue
            pairs.append(
                CollisionPair(
                    first_id=first_id,
                    second_id=second_id,
                    first=first,
                    second=second,
                )
            )
        return tuple(pairs)
