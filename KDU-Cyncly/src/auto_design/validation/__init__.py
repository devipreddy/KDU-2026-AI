"""Constraint validation package."""

from auto_design.validation.rules import (
    RuleViolation,
    VariantValidationResult,
    flatten_validation_results,
    validate_variant,
    validate_variants,
)

__all__ = [
    "RuleViolation",
    "VariantValidationResult",
    "flatten_validation_results",
    "validate_variant",
    "validate_variants",
]
