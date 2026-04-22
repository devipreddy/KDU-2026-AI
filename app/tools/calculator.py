from __future__ import annotations

import re
from typing import Any

import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from app.tools.base import BaseTool, ToolExecutionError


SAFE_CHARS_RE = re.compile(r"^[0-9A-Za-z\s\+\-\*/\^\(\)\.,%!]+$")
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate arithmetic, scientific, and symbolic math expressions."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate.",
            },
            "precision": {
                "type": "integer",
                "description": "Decimal precision for approximate output.",
                "default": 12,
                "minimum": 2,
                "maximum": 30,
            },
        },
        "required": ["expression"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.locals = {
            "sin": sympy.sin,
            "cos": sympy.cos,
            "tan": sympy.tan,
            "asin": sympy.asin,
            "acos": sympy.acos,
            "atan": sympy.atan,
            "sqrt": sympy.sqrt,
            "log": sympy.log,
            "ln": sympy.log,
            "exp": sympy.exp,
            "pi": sympy.pi,
            "e": sympy.E,
            "factorial": sympy.factorial,
            "abs": sympy.Abs,
        }

    async def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        expression = str(arguments.get("expression", "")).strip()
        precision = int(arguments.get("precision", 12))
        if not expression:
            raise ToolExecutionError("Calculator expression cannot be empty.")
        if not SAFE_CHARS_RE.match(expression) or any(
            token in expression for token in ("__", "[", "]", "{", "}", ";")
        ):
            raise ToolExecutionError("Expression contains unsupported characters.")

        normalized = expression.replace("^", "**")
        try:
            parsed = parse_expr(
                normalized,
                local_dict=self.locals,
                transformations=TRANSFORMATIONS,
                evaluate=True,
            )
            simplified = sympy.simplify(parsed)
            approximated = sympy.N(simplified, precision)
        except Exception as exc:
            raise ToolExecutionError(f"Failed to evaluate expression: {exc}") from exc

        return {
            "status": "success",
            "expression": expression,
            "exact_result": str(simplified),
            "approximate_result": str(approximated),
            "is_numeric": bool(simplified.is_number),
        }
