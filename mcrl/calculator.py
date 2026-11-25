"""Safe, bounded arithmetic extraction for simple natural-language requests.

Only numeric literals and whitelisted arithmetic AST nodes are evaluated. This
module never calls eval/exec and is independent of the topic knowledge store.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


_NUMBER = r"(-?\d+(?:\.\d+)?)"
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


@dataclass(frozen=True)
class Calculation:
    expression: str
    result: int | float
    operation: str

    def text(self) -> str:
        value = str(self.result)
        return f"{self.expression} = {value}."


def _number(value: str) -> Decimal:
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid number") from exc
    if not number.is_finite() or abs(number) > Decimal("1000000000"):
        raise ValueError("number outside supported range")
    return number


def _evaluate(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return _number(str(node.value))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("exponent too large")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("division by zero")
        result = _BINARY[type(node.op)](left, right)
        if not result.is_finite() or abs(result) > Decimal("1000000000000"):
            raise ValueError("result outside supported range")
        return result
    raise ValueError("unsupported arithmetic syntax")


def _format_decimal(value: Decimal) -> int | float:
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return int(normalized)
    return float(normalized)


def _extract_expression(text: str) -> tuple[str, str] | None:
    lowered = " ".join((text or "").lower().split())

    natural_patterns = (
        (rf"(?:sum|add|addition)\s+of\s+{_NUMBER}\s+(?:and|plus)\s+{_NUMBER}", "addition"),
        (rf"(?:subtract|difference)\s+(?:of\s+)?{_NUMBER}\s+from\s+{_NUMBER}", "subtraction"),
        (rf"(?:product|multiply|multiplication)\s+(?:of\s+)?{_NUMBER}\s+(?:and|by|times)\s+{_NUMBER}", "multiplication"),
    )
    for pattern, operation_name in natural_patterns:
        match = re.search(pattern, lowered)
        if match:
            first, second = match.groups()
            if operation_name == "subtraction" and lowered[match.start():match.end()].startswith(("subtract", "difference")):
                expression = f"{second} - {first}"
            elif operation_name == "multiplication":
                expression = f"{first} * {second}"
            else:
                expression = f"{first} + {second}"
            return expression, operation_name

    numeric = re.search(
        rf"(?<![A-Za-z0-9]){_NUMBER}\s*(\+|-|\*|/|plus|minus|times|multiplied\s+by|divided\s+by)\s*{_NUMBER}(?![A-Za-z0-9])",
        lowered,
    )
    if numeric:
        first, symbol, second = numeric.groups()
        mapping = {
            "+": ("+", "addition"), "plus": ("+", "addition"),
            "-": ("-", "subtraction"), "minus": ("-", "subtraction"),
            "*": ("*", "multiplication"), "times": ("*", "multiplication"),
            "multiplied by": ("*", "multiplication"),
            "/": ("/", "division"), "divided by": ("/", "division"),
        }
        operator_symbol, operation_name = mapping[symbol]
        return f"{first} {operator_symbol} {second}", operation_name

    return None


def try_calculate(text: str) -> Calculation | None:
    """Return a result only when an explicit, supported arithmetic form exists."""
    extracted = _extract_expression(text)
    if extracted is None:
        return None
    expression, operation = extracted
    try:
        tree = ast.parse(expression, mode="eval")
        value = _evaluate(tree)
    except (SyntaxError, ValueError, ArithmeticError, OverflowError):
        return None
    return Calculation(expression=expression, result=_format_decimal(value), operation=operation)
