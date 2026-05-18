"""Structured JSON contracts for the auto-design system."""

from auto_design.schemas.catalog import Product, ProductCatalog, ProductConstraints
from auto_design.schemas.common import DimensionsMM, HexColor, Point3D, WallDimensions
from auto_design.schemas.environment import Environment, Floor, Opening, Wall
from auto_design.schemas.input import DesignInput, Preferences
from auto_design.schemas.intent import (
    ColorRequest,
    MaterialRequest,
    PromptConstraint,
    StructuredIntent,
)
from auto_design.schemas.output import (
    LayoutItem,
    LayoutResponse,
    LayoutVariant,
    Rationale,
    Violation,
)

__all__ = [
    "ColorRequest",
    "DesignInput",
    "DimensionsMM",
    "Environment",
    "Floor",
    "HexColor",
    "LayoutItem",
    "LayoutResponse",
    "LayoutVariant",
    "MaterialRequest",
    "Opening",
    "Point3D",
    "Preferences",
    "Product",
    "ProductCatalog",
    "ProductConstraints",
    "PromptConstraint",
    "Rationale",
    "StructuredIntent",
    "Violation",
    "Wall",
    "WallDimensions",
]
