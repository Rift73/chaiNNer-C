"""Generate fixed C++ control flow from the pinned ONNX-to-NCNN application.

This is a development tool, never a runtime interpreter. Unsupported syntax
fails generation. Native numeric helpers implement array work; protobuf objects
retain their public native storage and original alias/exception semantics.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from graph_codegen_common import Emitter

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/tests/reference_onnx_graph/onnx_to_ncnn.py"
OUTPUT = ROOT / "native/include/onnx_converter_passes.hpp"
# The bridge's home sets its ruff settings; BRIDGE is where it is read and written.
BRIDGE_HOME = ROOT / "backend/src/nodes/impl/onnx/onnx_to_ncnn.py"
BRIDGE = BRIDGE_HOME
GLOBAL_READ = re.compile(r'item\(globals, py::str\("(\w+)"\)\)')
EXPORT_COMMENT = "# `x as x` imports: the native mirror reads these names in "


class OnnxEmitter(Emitter):
    def __init__(self) -> None:
        super().__init__()
        self.locals: set[str] = set()

    def ordered(self, values: list[str], render: Callable[[list[str]], str]) -> str:
        names = [self.uid() for _ in values]
        assignments = " ".join(
            f"O {name} = {value};" for name, value in zip(names, values, strict=True)
        )
        return "([&]() -> O { " + assignments + " return " + render(names) + "; }())"

    def cond(self, n: ast.AST) -> str:
        node = n
        if isinstance(node, ast.Compare):
            assert len(node.ops) == 1
            a, b = self.expr(node.left), self.expr(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, (ast.In, ast.NotIn)):
                return (
                    "truth("
                    + self.ordered(
                        [a, b],
                        lambda v: (
                            "py::bool_("
                            + ("!" if isinstance(op, ast.NotIn) else "")
                            + f"contains({v[1]}, {v[0]}))"
                        ),
                    )
                    + ")"
                )
            if isinstance(op, (ast.Is, ast.IsNot)):
                return (
                    "truth("
                    + self.ordered(
                        [a, b],
                        lambda v: (
                            f"py::bool_({v[0]}.ptr() {'!=' if isinstance(op, ast.IsNot) else '=='} {v[1]}.ptr())"
                        ),
                    )
                    + ")"
                )
            return "truth(" + self.expr(node) + ")"
        return super().cond(node)

    def expr(self, n: ast.AST) -> str:
        node = n
        if isinstance(node, ast.Name):
            if node.id == "self":
                return "self"
            if node.id in self.locals:
                return f'local(v_{node.id}, "{node.id}")'
            if node.id in {
                "int",
                "float",
                "list",
                "bool",
                "range",
                "isinstance",
                "enumerate",
                "len",
            }:
                return f'builtin("{node.id}")'
            return f'item(globals, py::str("{node.id}"))'
        if isinstance(node, ast.Attribute):
            return f'attr({self.expr(node.value)}, "{node.attr}")'
        if isinstance(node, ast.Subscript):
            return self.ordered(
                [self.expr(node.value), self.expr(node.slice)],
                lambda v: f"item({v[0]}, {v[1]})",
            )
        if isinstance(node, ast.BinOp):
            ops: dict[type[ast.operator], str] = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.Div: "div",
                ast.FloorDiv: "floordiv",
                ast.Pow: "pow",
                ast.Mod: "mod",
            }
            return self.ordered(
                [self.expr(node.left), self.expr(node.right)],
                lambda v: f"binary({v[0]}, {v[1]}, Op::{ops[type(node.op)]})",
            )
        if isinstance(node, ast.Constant) and isinstance(node.value, bytes):
            escaped = "".join(f"\\x{byte:02x}" for byte in node.value)
            return f'py::bytes("{escaped}", {len(node.value)})'
        if isinstance(node, ast.Compare):
            assert len(node.ops) == 1
            if isinstance(node.ops[0], (ast.In, ast.NotIn, ast.Is, ast.IsNot)):
                return "py::bool_(" + self.cond(node) + ")"
            codes: dict[type[ast.cmpop], str] = {
                ast.Eq: "Py_EQ",
                ast.NotEq: "Py_NE",
                ast.Lt: "Py_LT",
                ast.Gt: "Py_GT",
                ast.LtE: "Py_LE",
                ast.GtE: "Py_GE",
            }
            return self.ordered(
                [self.expr(node.left), self.expr(node.comparators[0])],
                lambda v: f"rich_compare({v[0]}, {v[1]}, {codes[type(node.ops[0])]})",
            )
        if isinstance(node, ast.BoolOp):
            parts = [self.expr(n) for n in node.values]
            temp = self.uid()
            body = f"O {temp} = {parts[0]}; "
            for part in parts[1:]:
                condition = (
                    f"truth({temp})"
                    if isinstance(node.op, ast.And)
                    else f"!truth({temp})"
                )
                body += f"if ({condition}) {temp} = {part}; "
            return "([&]() -> O { " + body + f"return {temp}; " + "}())"
        if isinstance(node, ast.JoinedStr):
            parts = []
            for part in node.values:
                if isinstance(part, ast.Constant):
                    parts.append(self.expr(part))
                else:
                    assert (
                        isinstance(part, ast.FormattedValue)
                        and part.format_spec is None
                        and part.conversion == -1
                    )
                    parts.append(f"py::str({self.expr(part.value)})")
            return "concat_text({" + ", ".join(parts) + "})"
        if isinstance(node, ast.Dict):
            return (
                "make_dict({"
                + ", ".join(
                    "{" + self.expr(cast(ast.AST, k)) + ", " + self.expr(v) + "}"
                    for k, v in zip(node.keys, node.values, strict=True)
                )
                + "})"
            )
        if isinstance(node, (ast.List, ast.Tuple)) and any(
            isinstance(n, ast.Starred) for n in node.elts
        ):
            name = self.uid()
            body = f"py::list {name}; "
            for element in node.elts:
                if isinstance(element, ast.Starred):
                    body += f'{name}.attr("extend")({self.expr(element.value)}); '
                else:
                    body += f"{name}.append({self.expr(element)}); "
            result = f"py::tuple({name})" if isinstance(node, ast.Tuple) else name
            return "([&]() -> O { " + body + f"return {result}; " + "}())"
        if isinstance(node, (ast.ListComp, ast.DictComp)):
            assert len(node.generators) == 1
            comp = node.generators[0]
            assert isinstance(comp.target, ast.Name) and not comp.is_async
            value = "v_" + comp.target.id
            self.locals.add(comp.target.id)
            result, cursor = self.uid(), self.uid()
            kind = "py::dict" if isinstance(node, ast.DictComp) else "py::list"
            body = f"{kind} {result}; for (py::handle {cursor} : py::reinterpret_borrow<py::iterable>({self.expr(comp.iter)})) {{ O {value} = py::reinterpret_borrow<O>({cursor}); "
            for check in comp.ifs:
                body += "if (!(" + self.cond(check) + ")) continue; "
            if isinstance(node, ast.DictComp):
                body += f"{result}[{self.expr(node.key)}] = {self.expr(node.value)}; "
            else:
                body += f"{result}.append({self.expr(node.elt)}); "
            return "([&]() -> O { " + body + "} return " + result + "; }())"
        if isinstance(node, ast.Call):
            args = [self.expr(a) for a in node.args]
            args.extend(self.expr(k.value) for k in node.keywords)

            def render_args(values: list[str]) -> list[str]:
                return values[: len(node.args)] + [
                    f'py::arg("{k.arg}") = {value}'
                    for k, value in zip(
                        node.keywords, values[len(node.args) :], strict=True
                    )
                ]

            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                owner, name = node.func.value.id, node.func.attr
                if owner == "self":
                    static = name in {"add_weight", "clear_container"}
                    prefix = ["globals"] + ([] if static else ["self"])
                    return self.ordered(
                        args,
                        lambda v: (
                            f"converter_{name}("
                            + ", ".join(prefix + render_args(v))
                            + ")"
                        ),
                    )
                if owner == "np":
                    target = {
                        "any": "array_any",
                        "all": "array_all",
                        "array": "array_construct",
                        "empty": "array_empty",
                        "sum": "array_sum",
                        "delete": "array_delete",
                    }.get(name)
                    assert target is not None, ast.unparse(node)
                    return self.ordered(
                        args, lambda v: target + "(" + ", ".join(render_args(v)) + ")"
                    )
            if isinstance(node.func, ast.Name) and node.func.id == "len":
                return "py::int_(py::len(" + args[0] + "))"
            return self.ordered(
                [self.expr(node.func), *args],
                lambda v: v[0] + "(" + ", ".join(render_args(v[1:])) + ")",
            )
        return super().expr(node)

    def assign(self, target: ast.AST, value: str) -> None:
        if isinstance(target, ast.Name):
            self.line(f"v_{target.id} = {value};")
        elif isinstance(target, (ast.Tuple, ast.List)):
            temp = self.uid()
            self.line(f"O {temp} = unpack({value}, {len(target.elts)});")
            for index, name in enumerate(target.elts):
                self.assign(name, f"item({temp}, py::int_({index}))")
        else:
            temp = self.uid()
            self.line(f"O {temp} = {value};")
            if isinstance(target, ast.Attribute):
                self.line(
                    f'set_attr({self.expr(target.value)}, "{target.attr}", {temp});'
                )
            elif isinstance(target, ast.Subscript):
                owner, key = self.uid(), self.uid()
                self.line(f"O {owner} = {self.expr(target.value)};")
                self.line(f"O {key} = {self.expr(target.slice)};")
                self.line(f"set_item({owner}, {key}, {temp});")
            else:
                raise ValueError(ast.dump(target))

    def statements(self, statements: list[ast.stmt]) -> None:
        for node in statements:
            if isinstance(node, ast.AnnAssign):
                if node.value is not None:
                    self.assign(node.target, self.expr(node.value))
            elif isinstance(node, ast.AugAssign):
                ops: dict[type[ast.operator], str] = {ast.Add: "add", ast.Sub: "sub"}
                op = ops[type(node.op)]
                owner = key = None
                if isinstance(node.target, ast.Subscript):
                    owner, key = self.uid(), self.uid()
                    self.line(f"O {owner} = {self.expr(node.target.value)};")
                    self.line(f"O {key} = {self.expr(node.target.slice)};")
                    original = f"item({owner}, {key})"
                elif isinstance(node.target, ast.Attribute):
                    owner = self.uid()
                    self.line(f"O {owner} = {self.expr(node.target.value)};")
                    original = f'attr({owner}, "{node.target.attr}")'
                else:
                    assert isinstance(node.target, ast.Name)
                    original = self.expr(node.target)
                left, right, result = self.uid(), self.uid(), self.uid()
                self.line(f"O {left} = {original};")
                self.line(f"O {right} = {self.expr(node.value)};")
                self.line(f"O {result} = inplace({left}, {right}, Op::{op});")
                if isinstance(node.target, ast.Subscript):
                    self.line(f"set_item({owner}, {key}, {result});")
                elif isinstance(node.target, ast.Attribute):
                    self.line(f'set_attr({owner}, "{node.target.attr}", {result});')
                else:
                    self.assign(node.target, result)
            elif isinstance(node, ast.Assign) and len(node.targets) > 1:
                temp = self.uid()
                self.line(f"O {temp} = {self.expr(node.value)};")
                for target in node.targets:
                    self.assign(target, temp)
            elif isinstance(node, ast.Return):
                self.line(
                    "return "
                    + (self.expr(node.value) if node.value else "py::none()")
                    + ";"
                )
            elif isinstance(node, ast.Raise):
                assert (
                    isinstance(node.exc, ast.Call)
                    and isinstance(node.exc.func, ast.Name)
                    and node.cause is None
                )
                assert len(node.exc.args) == 1
                self.line(
                    f"raise(PyExc_{node.exc.func.id}, {self.expr(node.exc.args[0])});"
                )
            elif isinstance(node, ast.Assert):
                if node.msg is None:
                    self.line(
                        f"if (!({self.cond(node.test)})) {{ PyErr_SetNone(PyExc_AssertionError); throw py::error_already_set(); }}"
                    )
                else:
                    self.line(
                        f"if (!({self.cond(node.test)})) raise(PyExc_AssertionError, {self.expr(node.msg)});"
                    )
            elif isinstance(node, ast.If):
                branch = node  # walks the nested elif chain
                self.line("if (" + self.cond(branch.test) + ") {")
                while True:
                    self.indent += 1
                    self.statements(branch.body)
                    self.indent -= 1
                    if len(branch.orelse) == 1 and isinstance(branch.orelse[0], ast.If):
                        branch = branch.orelse[0]
                        self.line("} else if (" + self.cond(branch.test) + ") {")
                        continue
                    if branch.orelse:
                        self.line("} else {")
                        self.indent += 1
                        self.statements(branch.orelse)
                        self.indent -= 1
                    self.line("}")
                    break
            elif isinstance(node, ast.For):
                if (
                    node.body
                    and isinstance(node.body[-1], ast.Raise)
                    and not node.orelse
                ):
                    cursor = self.uid()
                    self.line(f"auto {cursor} = py::iter({self.expr(node.iter)});")
                    self.line(f"if ({cursor} != py::iterator::sentinel()) {{")
                    self.indent += 1
                    self.assign(node.target, f"py::reinterpret_borrow<O>(*{cursor})")
                    self.statements(node.body)
                    self.indent -= 1
                    self.line("}")
                    continue
                flag = self.uid() if node.orelse else None
                if flag:
                    self.line(f"bool {flag} = false;")
                self.loops.append(flag)
                cursor = self.uid()
                self.line(
                    f"for (py::handle {cursor} : py::reinterpret_borrow<py::iterable>({self.expr(node.iter)})) {{"
                )
                self.indent += 1
                self.assign(node.target, f"py::reinterpret_borrow<O>({cursor})")
                self.statements(node.body)
                self.indent -= 1
                self.line("}")
                self.loops.pop()
                if flag:
                    self.line(f"if (!{flag}) {{")
                    self.indent += 1
                    self.statements(node.orelse)
                    self.indent -= 1
                    self.line("}")
            else:
                super().statements([node])

    def function(self, f: ast.FunctionDef, declaration: bool = False) -> None:
        fn = f
        arguments = [arg.arg for arg in fn.args.args]
        params = ["const O &globals"] + [
            ("const O &self" if a == "self" else "O v_" + a) for a in arguments
        ]
        if declaration and fn.args.defaults:
            for index, value in enumerate(
                fn.args.defaults, start=len(params) - len(fn.args.defaults)
            ):
                params[index] += " = " + self.expr(value)
        name = fn.name.strip("_")
        signature = "O converter_" + name + "(" + ", ".join(params) + ")"
        if declaration:
            self.line(signature + ";")
            return
        self.locals = {
            n.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        } | set(arguments)
        self.line(f"// Frozen installed onnx_to_ncnn.py:{fn.lineno}")
        self.line(signature + " {")
        self.indent += 1
        for name in sorted(self.locals - set(arguments)):
            self.line(f"O v_{name};")
        self.line("(void)globals;")
        self.statements(fn.body)
        if not isinstance(fn.body[-1], ast.Return):
            self.line("return py::none();")
        self.indent -= 1
        self.line("}")


def global_reads(lines: list[str]) -> set[str]:
    """The names emitted C++ reads from its bridge module's globals."""
    return {name for line in lines for name in GLOBAL_READ.findall(line)}


def ruff(args: list[str], text: str, config_path: Path) -> str:
    """Run the environment's ruff on text; config_path selects the repo settings."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            *args,
            "--stdin-filename",
            str(config_path),
            "-",
        ],
        input=text,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def write_bridge(
    path: Path, text: str, reads: set[str], consumer: str, config_path: Path
) -> None:
    """Write a lint-clean bridge that keeps the globals native code reads.

    Each import binding the emitted C++ reads takes the PEP 484 explicit-export
    form (`x as x`) under one comment naming the consumer, so nothing drops it.
    The repo's ruff settings, found from config_path, sort and format the result.
    """
    lines = [line for line in text.splitlines() if not line.startswith(EXPORT_COMMENT)]
    edits = [
        (alias.end_lineno - 1, alias.end_col_offset, f" as {alias.name}")
        for node in ast.parse("\n".join(lines)).body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
        if alias.asname is None
        and "." not in alias.name
        and alias.name in reads
        and alias.end_lineno is not None
        and alias.end_col_offset is not None
    ]
    for row, column, insertion in sorted(edits, reverse=True):
        lines[row] = lines[row][:column] + insertion + lines[row][column:]
    # Generated text carries no meaningful trailing commas, so imports that fit
    # are joined onto one line.
    text = ruff(
        [
            "check",
            "--select",
            "I",
            "--fix",
            "--exit-zero",
            "--quiet",
            "--config",
            "lint.isort.split-on-trailing-comma = false",
        ],
        "\n".join(lines) + "\n",
        config_path,
    )
    exports = [
        node.lineno
        for node in ast.parse(text).body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(alias.asname == alias.name for alias in node.names)
    ]
    if exports:
        lines = text.splitlines()
        lines.insert(exports[0] - 1, EXPORT_COMMENT + consumer)
        text = "\n".join(lines) + "\n"
    # LF on every platform: backend/src is -text, so these are the bytes every
    # checkout gets, in upstream's line ending.
    path.write_text(ruff(["format"], text, config_path), encoding="utf-8", newline="\n")


def main() -> None:
    data = SOURCE.read_bytes()
    manifest = json.loads(SOURCE.with_name("sources.json").read_text())
    assert hashlib.sha256(data).hexdigest() == manifest["files"][SOURCE.name]
    tree = ast.parse(data)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    functions = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
    emitter = OnnxEmitter()
    emitter.line(
        "// Fixed C++ conversion; generated by native/tools/generate_onnx_converter.py."
    )
    emitter.line("// Source SHA256: " + hashlib.sha256(data).hexdigest())
    for fn in functions:
        emitter.function(fn, declaration=True)
    for fn in functions:
        emitter.function(fn)
    emitter.line("void bind_converter(py::module_ &module) {")
    for fn in functions:
        emitter.line(
            f'    module.def("onnx_converter_{fn.name.strip("_")}", &converter_{fn.name.strip("_")});'
        )
    emitter.line("}")
    OUTPUT.write_text("\n".join(emitter.lines) + "\n", encoding="utf-8", newline="\n")
    # Preserve the original public signatures and annotations. Only execution
    # bodies change; the generated C++ never imports this development tool.
    header = BRIDGE.read_text(encoding="utf-8").split("class Onnx2NcnnConverter:")[0]
    if "from ..native_graph import graph" not in header:
        header += "from ..native_graph import graph\n\n\n"
    bridge = [header + "class Onnx2NcnnConverter:"]
    for fn in functions:
        wrapper = copy.deepcopy(fn)
        args: list[ast.expr] = [
            ast.Call(func=ast.Name(id="globals", ctx=ast.Load()), args=[], keywords=[])
        ]
        args.extend(ast.Name(id=a.arg, ctx=ast.Load()) for a in fn.args.args)
        call = ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id="graph", ctx=ast.Load()), args=[], keywords=[]
                ),
                attr="onnx_converter_" + fn.name.strip("_"),
                ctx=ast.Load(),
            ),
            args=args,
            keywords=[],
        )
        wrapper.body = [ast.Expr(call) if fn.name == "__init__" else ast.Return(call)]
        ast.fix_missing_locations(wrapper)
        bridge.append(
            "\n".join(
                "    " + line if line else ""
                for line in ast.unparse(wrapper).splitlines()
            )
        )
        bridge.append("")
    write_bridge(
        BRIDGE,
        "\n".join(bridge) + "\n",
        global_reads(emitter.lines),
        "native/include/onnx_converter_passes.hpp",
        BRIDGE_HOME,
    )
    print(f"Generated {len(functions)} complete native converter functions.")


if __name__ == "__main__":
    main()
