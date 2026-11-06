from __future__ import annotations

import ast
import math
import operator as op
from typing import Any, ClassVar

from nodes.groups import optional_list_group
from nodes.properties.inputs import NumberInput, TextInput
from nodes.properties.outputs import NumberOutput
from nodes.utils.utils import ALPHABET

from .. import math_group

# Parsing's memory follows the text: 20 MB of it take 3 GB.
MAX_EXPRESSION_LENGTH = 4096
# An error shows no more than this of the expression, and of what it says about it.
SHOWN_TEXT_LENGTH = 200


def shown_text(text: str) -> str:
    if len(text) <= SHOWN_TEXT_LENGTH:
        return text
    return f"{text[:SHOWN_TEXT_LENGTH]}..."


def round_to_digits(value: float, digits: float | None = None) -> float:
    """Python's round, for the float that every number of an expression is."""
    if digits is None:
        return round(value)
    if not digits.is_integer():
        raise ValueError(f"round takes a whole number of digits, not {digits}")
    return round(value, int(digits))


class SafeMathEvaluator:
    """Safe evaluator for mathematical expressions using AST parsing."""

    # Supported operators
    operators: ClassVar[dict[type, Any]] = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv,
        ast.Pow: op.pow,
        ast.Mod: op.mod,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    # Supported functions
    functions: ClassVar[dict[str, Any]] = {
        "abs": abs,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "round": round_to_digits,
        "min": min,
        "max": max,
        "pow": pow,
        "radians": math.radians,
        "degrees": math.degrees,
        # Constants
        "pi": math.pi,
        "e": math.e,
    }

    def __init__(self, variables: dict[str, float]):
        self.variables = variables

    @staticmethod
    def not_nan(value: float) -> float:
        # min and max would keep or drop a NaN by the order of their arguments.
        if math.isnan(value):
            raise ValueError("A step of the expression is not a number")
        return value

    def eval_node(self, node: Any) -> float:
        if isinstance(node, ast.Constant):
            # Ensure we only handle numeric constants
            if isinstance(node.value, int | float) and not isinstance(node.value, bool):
                return float(node.value)
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.Name):
            if node.id in self.variables:
                # An integer power of integers could take any time and memory.
                return self.not_nan(float(self.variables[node.id]))
            elif node.id in self.functions:
                # For constants like pi, e
                value = self.functions[node.id]
                if not callable(value):
                    return float(value)
                raise NameError(f"Name '{node.id}' requires arguments")
            raise NameError(f"Name '{node.id}' is not defined")
        elif isinstance(node, ast.BinOp):
            left = self.eval_node(node.left)
            right = self.eval_node(node.right)
            op_func = self.operators.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Operator {type(node.op).__name__} is not supported")
            result = op_func(left, right)
            # e.g. a negative number to a fractional power
            if isinstance(result, complex):
                raise ValueError("The result is not a real number")
            return self.not_nan(result)
        elif isinstance(node, ast.UnaryOp):
            operand = self.eval_node(node.operand)
            op_func = self.operators.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Operator {type(node.op).__name__} is not supported")
            return op_func(operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name is None or func_name not in self.functions:
                raise ValueError(f"Function '{func_name}' is not supported")
            func_obj = self.functions[func_name]
            if not callable(func_obj):
                raise ValueError(f"'{func_name}' is a constant, not a function")
            if node.keywords:
                raise ValueError(f"Function '{func_name}' takes no named arguments")
            args = [self.eval_node(arg) for arg in node.args]
            result = func_obj(*args)
            # Ensure result is a float
            if isinstance(result, int | float):
                return self.not_nan(float(result))
            raise ValueError(
                f"Function '{func_name}' returned non-numeric value: {type(result)}"
            )
        else:
            raise ValueError(f"Node type {type(node).__name__} is not supported")

    def eval(self, expr: str) -> float:
        if len(expr) > MAX_EXPRESSION_LENGTH:
            raise ValueError(
                f"The expression is longer than {MAX_EXPRESSION_LENGTH} characters"
            )
        try:
            # Python's grammar takes a leading space for an indent.
            node = ast.parse(expr.strip(), mode="eval").body
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}") from e
        result = self.eval_node(node)
        # A later node would fail on it, with no word of where it came from.
        if not math.isfinite(result):
            raise ValueError(f"The result is {result}, not a finite number")
        return result


@math_group.register(
    schema_id="chainner:utility:math_expression",
    name="Math Expression",
    description=[
        "Evaluate a mathematical expression with up to 26 variables (a-z).",
        "Supports common mathematical operations (+, -, *, /, **, %, //) and functions (sqrt, sin, cos, log, etc.).",
        "Example: `sqrt(a**2 + b**2) * c / 100`",
    ],
    icon="MdCalculate",
    inputs=[
        TextInput(
            "Expression",
            placeholder="sqrt(a**2 + b**2)",
            multiline=False,
            min_length=1,
        ).with_id(0),
        optional_list_group(
            *[
                NumberInput(
                    letter.lower(), min=None, max=None, precision="unlimited", step=1
                )
                .with_id(1 + i)
                .make_optional()
                for i, letter in enumerate(ALPHABET)
            ]
        ),
    ],
    outputs=[
        NumberOutput(
            "Result",
            output_type="number",
        )
    ],
)
def math_expression_node(expression: str, *args: float | None) -> float:
    # Build variable dictionary from provided inputs
    variables: dict[str, float] = {}
    alphabet_lower = [letter.lower() for letter in ALPHABET[:26]]

    for i, value in enumerate(args):
        if value is not None:
            variables[alphabet_lower[i]] = value

    # Evaluate the expression (variables can be empty for constant expressions)
    evaluator = SafeMathEvaluator(variables)
    try:
        result = evaluator.eval(expression)
        return result
    except Exception as e:
        raise ValueError(
            f"Error evaluating expression '{shown_text(expression)}': {shown_text(str(e))}"
        ) from e
