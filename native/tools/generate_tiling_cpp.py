"""Generate fixed C++ tile state machines from pinned installed/source variants.

The two public callback/progress APIs remain separate. Only the installed
variant ships: its Python bridges are written into backend/src. The source
variant's functions stay in tiling_control.hpp because the shipped native
module still binds them. This development tool never runs in the application
and rejects unsupported syntax.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

from generate_onnx_converter import OnnxEmitter, global_reads, write_bridge

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "native/tests/reference_tiling"
OUTPUT = ROOT / "native/include/tiling_control.hpp"
# The bridges' home sets their ruff settings; BRIDGES is where they are written.
BRIDGE_HOME = ROOT / "backend/src/nodes/impl/upscale"
BRIDGES = BRIDGE_HOME


class TilingEmitter(OnnxEmitter):
    def expr(self, n: ast.AST) -> str:
        node = n
        if isinstance(node, ast.Name) and node.id in {"min", "max", "sum"}:
            return f'builtin("{node.id}")'
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sum"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.GeneratorExp)
        ):
            generator = node.args[0]
            assert len(generator.generators) == 1
            comp = generator.generators[0]
            assert (
                isinstance(comp.target, ast.Name) and not comp.ifs and not comp.is_async
            )
            self.locals.add(comp.target.id)
            total, cursor = self.uid(), self.uid()
            return (
                "([&]() -> O { O "
                + total
                + " = py::int_(0); for(py::handle "
                + cursor
                + " : py::reinterpret_borrow<py::iterable>("
                + self.expr(comp.iter)
                + ")) { O v_"
                + comp.target.id
                + " = py::reinterpret_borrow<O>("
                + cursor
                + "); "
                + total
                + " = binary("
                + total
                + ", "
                + self.expr(generator.elt)
                + ", Op::add); } return "
                + total
                + "; }())"
            )
        if isinstance(node, ast.Call) and any(
            isinstance(arg, ast.Starred) for arg in node.args
        ):
            assert not node.keywords
            function, arguments = self.uid(), self.uid()
            body = f"O {function} = {self.expr(node.func)}; py::list {arguments}; "
            for arg in node.args:
                if isinstance(arg, ast.Starred):
                    body += f'{arguments}.attr("extend")({self.expr(arg.value)}); '
                else:
                    body += f"{arguments}.append({self.expr(arg)}); "
            return (
                "([&]() -> O { "
                + body
                + f"return {function}(*py::tuple({arguments})); "
                + "}())"
            )
        return super().expr(node)

    def statements(self, statements: list[ast.stmt]) -> None:
        for node in statements:
            if isinstance(node, ast.FunctionDef):
                arguments = {arg.arg for arg in node.args.args}
                stores = {
                    n.id
                    for n in ast.walk(node)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                }
                previous = self.locals.copy()
                self.locals |= arguments | stores | {node.name}
                params = ", ".join("O v_" + arg.arg for arg in node.args.args)
                self.line(
                    f"v_{node.name} = py::cpp_function([=]({params}) mutable -> O {{"
                )
                self.indent += 1
                for name in sorted(stores - arguments):
                    self.line(f"O v_{name};")
                self.statements(node.body)
                if not isinstance(node.body[-1], ast.Return):
                    self.line("return py::none();")
                self.indent -= 1
                self.line("});")
                self.locals = previous | {node.name}
            elif isinstance(node, ast.Raise) and isinstance(node.exc, ast.Name):
                self.line(f"raise_class({self.expr(node.exc)});")
            elif isinstance(node, ast.Try):
                assert (
                    not node.finalbody and not node.orelse and len(node.handlers) == 1
                )
                handler = node.handlers[0]
                assert isinstance(handler.type, ast.Name) and handler.name is None
                self.line("try {")
                self.indent += 1
                self.statements(node.body)
                self.indent -= 1
                self.line("} catch (const py::error_already_set &error) {")
                self.indent += 1
                self.line(
                    f"if (!error.matches({self.expr(handler.type)}.ptr())) throw;"
                )
                self.line("GraphHandledException handled(error);")
                self.statements(handler.body)
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
        signature = "O tile_" + fn.name + "(" + ", ".join(params) + ")"
        if declaration:
            self.line(signature + ";")
            return
        self.locals = {
            n.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        } | set(arguments)
        self.locals |= {
            n.name
            for n in ast.walk(fn)
            if isinstance(n, ast.FunctionDef) and n is not fn
        }
        self.line(signature + " {")
        self.indent += 1
        for name in sorted(self.locals - set(arguments)):
            self.line(f"O v_{name};")
        self.line("(void)globals;")
        for name in arguments:
            self.line("(void)" + ("self" if name == "self" else "v_" + name) + ";")
        self.statements(fn.body)
        if not isinstance(fn.body[-1], (ast.Return, ast.Raise)):
            self.line("return py::none();")
        self.indent -= 1
        self.line("}")


def bridge(fn: ast.FunctionDef, name: str) -> ast.FunctionDef:
    result = copy.deepcopy(fn)
    args: list[ast.expr] = [
        ast.Call(func=ast.Name(id="globals", ctx=ast.Load()), args=[], keywords=[])
    ]
    args.extend(ast.Name(id=arg.arg, ctx=ast.Load()) for arg in fn.args.args)
    call = ast.Call(
        func=ast.Attribute(
            value=ast.Call(
                func=ast.Name(id="graph", ctx=ast.Load()), args=[], keywords=[]
            ),
            attr="tiling_" + name,
            ctx=ast.Load(),
        ),
        args=args,
        keywords=[],
    )
    result.body = [ast.Expr(call) if fn.name == "__init__" else ast.Return(call)]
    return ast.fix_missing_locations(result)


def main() -> None:
    paths = sorted(FROZEN.glob("*/*.py"))
    hashes = {
        str(path.relative_to(FROZEN)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }
    manifest = FROZEN / "sources.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == hashes
    else:
        manifest.write_text(
            json.dumps(hashes, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    emitter = TilingEmitter()
    emitter.line(
        "// Fixed native tile state machines. See generate_tiling_cpp.py and reference_tiling/sources.json."
    )
    functions = []
    for path in paths:
        variant = path.parent.name
        ships = variant == "installed"
        first = len(functions)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in tree.body:
            members = (
                [node]
                if isinstance(node, ast.FunctionDef)
                else node.body
                if isinstance(node, ast.ClassDef)
                else []
            )
            for index, fn in enumerate(members):
                if not isinstance(fn, ast.FunctionDef):
                    continue
                name = (
                    variant
                    + "_"
                    + path.stem
                    + "_"
                    + (node.name + "_" if isinstance(node, ast.ClassDef) else "")
                    + fn.name.strip("_")
                )
                native = copy.deepcopy(fn)
                native.name = name
                functions.append(native)
                if not ships:
                    continue
                if isinstance(node, ast.ClassDef):
                    node.body[index] = bridge(fn, name)
                else:
                    tree.body[tree.body.index(node)] = bridge(fn, name)
        if not ships:
            continue
        # Retain public classes, decorators, annotations and module-level imports.
        position = (
            1
            if isinstance(tree.body[0], ast.ImportFrom)
            and tree.body[0].module == "__future__"
            else 0
        )
        tree.body.insert(
            position,
            ast.ImportFrom(
                module="native_graph", names=[ast.alias(name="graph")], level=2
            ),
        )
        text = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
        # The module's own C functions name the globals its bridge must keep.
        scan = TilingEmitter()
        for fn in functions[first:]:
            scan.function(copy.deepcopy(fn))
        write_bridge(
            BRIDGES / path.name,
            text,
            global_reads(scan.lines),
            "native/include/tiling_control.hpp",
            BRIDGE_HOME / path.name,
        )
    for fn in functions:
        emitter.function(fn, declaration=True)
    for fn in functions:
        emitter.function(fn)
    emitter.line("void bind_tiling(py::module_ &module) {")
    for fn in functions:
        emitter.line(f'    module.def("tiling_{fn.name}", &tile_{fn.name});')
    emitter.line("}")
    OUTPUT.write_text("\n".join(emitter.lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Generated {len(functions)} fixed tiling functions across both pinned APIs.")


if __name__ == "__main__":
    main()
