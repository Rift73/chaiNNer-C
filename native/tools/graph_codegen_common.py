"""Translate the frozen installed passes to fixed C++ control flow.

This is a build-independent provenance tool, never imported by production.
Unsupported syntax stops generation rather than emitting a Python fallback.
Array expressions target the hand-written native kernels in ncnn_graph.cpp.
"""

import ast
import json


class Emitter:
    def __init__(self):
        self.lines = []
        self.indent = 0
        self.serial = 0
        self.loops = []

    def line(self, text: str):
        self.lines.append("    " * self.indent + text)

    def uid(self):
        self.serial += 1
        return f"cn_{self.serial}"

    def expr(self, n: ast.AST):
        if isinstance(n, ast.Constant):
            if n.value is None:
                return "py::none()"
            if isinstance(n.value, bool):
                return "py::bool_(" + str(n.value).lower() + ")"
            if isinstance(n.value, str):
                return "py::str(" + json.dumps(n.value) + ")"
            return (
                ("py::float_(" if isinstance(n.value, float) else "py::int_(")
                + repr(n.value)
                + ")"
            )
        if isinstance(n, ast.Name):
            if n.id == "self":
                return "model"
            if n.id in ("int", "float", "list", "bool"):
                return 'builtin("' + n.id + '")'
            if n.id == "NcnnLayer":
                return 'types.attr("NcnnLayer")'
            return n.id
        if isinstance(n, ast.Attribute):
            if (
                isinstance(n.value, ast.Name)
                and n.value.id == "self"
                and n.attr == "model"
            ):
                return "model"
            if isinstance(n.value, ast.Name) and n.value.id in ("BOT", "EOT"):
                return f'types.attr("{"BinaryOpTypes" if n.value.id == "BOT" else "EltwiseOpTypes"}").attr("{n.attr}")'
            if isinstance(n.value, ast.Name) and n.value.id == "np":
                return f'np().attr("{n.attr}")'
            return f'attr({self.expr(n.value)}, "{n.attr}")'
        if isinstance(n, ast.Subscript):
            return f"item({self.expr(n.value)}, {self.expr(n.slice)})"
        if isinstance(n, ast.Slice):
            vals = [
                self.expr(v) if v is not None else "py::none()"
                for v in (n.lower, n.upper, n.step)
            ]
            return "py::slice(" + ", ".join(vals) + ")"
        if isinstance(n, (ast.List, ast.Tuple)):
            return (
                ("make_list" if isinstance(n, ast.List) else "make_tuple")
                + "({"
                + ", ".join(self.expr(v) for v in n.elts)
                + "})"
            )
        if isinstance(n, ast.BinOp):
            ops: dict[type[ast.operator], str] = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.Div: "div",
                ast.FloorDiv: "floordiv",
                ast.Pow: "pow",
                ast.Mod: "mod",
            }
            return f"binary({self.expr(n.left)}, {self.expr(n.right)}, Op::{ops[type(n.op)]})"
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.Not):
                return f"py::bool_(!truth({self.expr(n.operand)}))"
            return f"negative({self.expr(n.operand)})"
        if isinstance(n, (ast.BoolOp, ast.Compare)):
            return "py::bool_(" + self.cond(n) + ")"
        if isinstance(n, ast.IfExp):
            return f"({self.cond(n.test)} ? O({self.expr(n.body)}) : O({self.expr(n.orelse)}))"
        if isinstance(n, ast.Call):
            args = ", ".join(self.expr(a) for a in n.args)
            if isinstance(n.func, ast.Name):
                name = n.func.id
                if name == "len":
                    return f"py::int_(py::len({args}))"
                if name == "checked_cast":
                    return "checked(" + args + ")"
                if name == "range":
                    return 'builtin("range")(' + args + ")"
                return self.expr(n.func) + "(" + args + ")"
            if isinstance(n.func, ast.Attribute):
                if (
                    isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "self"
                    and n.func.attr.startswith("__")
                ):
                    return "pass_" + n.func.attr[2:] + "(model, types)"
                if isinstance(n.func.value, ast.Name) and n.func.value.id == "np":
                    names = {
                        "sqrt": "array_sqrt",
                        "transpose": "array_transpose",
                        "broadcast_to": "array_broadcast",
                        "zeros": "array_zeros",
                        "ndarray": "array_empty",
                    }
                    if n.func.attr not in names:
                        raise ValueError(ast.dump(n))
                    kw = [self.expr(k.value) for k in n.keywords]
                    return (
                        names[n.func.attr]
                        + "("
                        + ", ".join([self.expr(a) for a in n.args] + kw)
                        + ")"
                    )
                if n.func.attr == "astype":
                    return "array_cast(" + self.expr(n.func.value) + ", " + args + ")"
                if n.func.attr == "reshape":
                    return (
                        "array_reshape(" + self.expr(n.func.value) + ", " + args + ")"
                    )
                return self.expr(n.func) + "(" + args + ")"
        raise ValueError(ast.dump(n))

    def cond(self, n: ast.AST):
        if isinstance(n, ast.BoolOp):
            op = " && " if isinstance(n.op, ast.And) else " || "
            return "(" + op.join(self.cond(v) for v in n.values) + ")"
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return "!(" + self.cond(n.operand) + ")"
        if isinstance(n, ast.Compare):
            assert len(n.ops) == 1
            a, b, op = self.expr(n.left), self.expr(n.comparators[0]), n.ops[0]
            if isinstance(op, (ast.In, ast.NotIn)):
                return (
                    "!" if isinstance(op, ast.NotIn) else ""
                ) + f"contains({b}, {a})"
            if isinstance(op, (ast.Is, ast.IsNot)):
                return f"({a}.ptr() {'!=' if isinstance(op, ast.IsNot) else '=='} {b}.ptr())"
            codes: dict[type[ast.cmpop], str] = {
                ast.Eq: "Py_EQ",
                ast.NotEq: "Py_NE",
                ast.Lt: "Py_LT",
                ast.Gt: "Py_GT",
                ast.LtE: "Py_LE",
                ast.GtE: "Py_GE",
            }
            code = codes[type(op)]
            return f"compare({a}, {b}, {code})"
        return "truth(" + self.expr(n) + ")"

    def assign(self, target: ast.AST, value: str):
        if isinstance(target, ast.Name):
            self.line(f"{target.id} = {value};")
        elif isinstance(target, ast.Attribute):
            self.line(f'set_attr({self.expr(target.value)}, "{target.attr}", {value});')
        elif isinstance(target, ast.Subscript):
            self.line(
                f"set_item({self.expr(target.value)}, {self.expr(target.slice)}, {value});"
            )
        else:
            raise ValueError(ast.dump(target))

    def statements(self, statements: list[ast.stmt]):
        for n in statements:
            if isinstance(n, ast.Expr):
                if not isinstance(n.value, ast.Constant):
                    self.line(self.expr(n.value) + ";")
            elif isinstance(n, ast.Assign):
                assert len(n.targets) == 1
                self.assign(n.targets[0], self.expr(n.value))
            elif isinstance(n, ast.AugAssign):
                ops: dict[type[ast.operator], str] = {ast.Add: "add", ast.Sub: "sub"}
                op = ops[type(n.op)]
                self.assign(
                    n.target,
                    f"inplace({self.expr(n.target)}, {self.expr(n.value)}, Op::{op})",
                )
            elif isinstance(n, ast.If):
                self.line("if (" + self.cond(n.test) + ") {")
                self.indent += 1
                self.statements(n.body)
                self.indent -= 1
                if n.orelse:
                    self.line("} else {")
                    self.indent += 1
                    self.statements(n.orelse)
                    self.indent -= 1
                self.line("}")
            elif isinstance(n, (ast.For, ast.While)):
                flag = self.uid() if n.orelse else None
                if flag:
                    self.line(f"bool {flag} = false;")
                self.loops.append(flag)
                if isinstance(n, ast.While):
                    self.line("while (" + self.cond(n.test) + ") {")
                    self.indent += 1
                else:
                    it = self.uid()
                    if (
                        isinstance(n.iter, ast.Call)
                        and isinstance(n.iter.func, ast.Name)
                        and n.iter.func.id == "enumerate"
                    ):
                        assert isinstance(n.target, ast.Tuple)
                        idx = self.uid()
                        self.line(f"py::ssize_t {idx} = 0;")
                        self.line(
                            f"for (py::handle {it} : py::reinterpret_borrow<py::iterable>({self.expr(n.iter.args[0])})) {{"
                        )
                        self.indent += 1
                        self.assign(n.target.elts[0], f"py::int_({idx}++)")
                        self.assign(
                            n.target.elts[1], f"py::reinterpret_borrow<O>({it})"
                        )
                    else:
                        self.line(
                            f"for (py::handle {it} : py::reinterpret_borrow<py::iterable>({self.expr(n.iter)})) {{"
                        )
                        self.indent += 1
                        self.assign(n.target, f"py::reinterpret_borrow<O>({it})")
                self.statements(n.body)
                self.indent -= 1
                self.line("}")
                self.loops.pop()
                if flag:
                    self.line(f"if (!{flag}) {{")
                    self.indent += 1
                    self.statements(n.orelse)
                    self.indent -= 1
                    self.line("}")
            elif isinstance(n, ast.Break):
                if self.loops[-1]:
                    self.line(self.loops[-1] + " = true;")
                self.line("break;")
            elif isinstance(n, ast.Continue):
                self.line("continue;")
            elif isinstance(n, ast.Try):
                assert not n.finalbody and not n.orelse and len(n.handlers) == 1
                handler = n.handlers[0]
                assert (
                    isinstance(handler.type, ast.Name) and handler.type.id == "KeyError"
                )
                self.line("try {")
                self.indent += 1
                self.statements(n.body)
                self.indent -= 1
                self.line("} catch (py::error_already_set &error) {")
                self.indent += 1
                self.line("if (!error.matches(PyExc_KeyError)) throw;")
                self.statements(handler.body)
                self.indent -= 1
                self.line("}")
            elif isinstance(n, ast.Pass):
                pass
            else:
                raise ValueError(ast.dump(n))

    def function(self, f: ast.FunctionDef):
        name = "pass_" + f.name.removeprefix("__")
        self.line(f"// Frozen installed optimizer.py:{f.lineno}")
        self.line(f"void {name}(const O &model, const O &types) {{")
        self.indent += 1
        names = sorted(
            {
                n.id
                for n in ast.walk(f)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            }
        )
        if names:
            self.line("O " + ", ".join(v + " = py::none()" for v in names) + ";")
        self.line("(void)types;")
        self.statements(f.body)
        self.indent -= 1
        self.line("}")
        self.line("")
