"""Local self-healing repairs for generated kitchen variants."""

from auto_design.repair.local import (
    RepairAction,
    RepairResult,
    flatten_repair_actions,
    flatten_repair_violations,
    repair_variant,
    repair_variants,
)

__all__ = [
    "RepairAction",
    "RepairResult",
    "flatten_repair_actions",
    "flatten_repair_violations",
    "repair_variant",
    "repair_variants",
]
