"""Temporal Fix's window part cut into pieces at its directly launched kernels.

A plugin's output reaches the next convolution only through a copy (TensorRT moves it into
the convolution's layout, and pads 25 or 26 channels to 32), and a convolution fed by a
plugin does not fuse with its bias and activation. An engine input declared HWC8 has
neither: the convolution reads it as it is. So the window's lowered graph (tfix_aot.lower)
is cut at the ops in tfix_aot.DIRECT: each piece is an engine, and each cut's kernel is
launched between two pieces, reading the earlier pieces' outputs and writing the next
piece's first input as rows of 32 (its channels, then zeros: HWC8's layout).

Build side (`cut`, `compile_kernels`): the pieces of a lowered model (None when no direct
op matched: the window part is then built whole, as before) and their kernels' cubins.
What a piece hands on: each kernel's inputs, and whatever a later piece reads. A later
piece recomputes rather than receives what is made of constants alone and the views,
slices and casts of the window's inputs; a tensor a later piece only slices is handed on
as its slices. Every other plugin goes to the piece that makes its outputs; a piece
gets a constant stand-in for each such output (dual_aot.attach replaces it).

Run side (`WindowPieces`): the pieces' engines and the kernels (their cubins loaded with
cuda.bindings on the session's device) as one call with the window part's bindings, whose
CUDA graph holds every engine and kernel of the call (temporal_session.capture_graph).

The layout (LAYOUT, JSON) and each kernel's cubin are parts of the engine set whose names
hold a dot (temporal_session.open_set does not take them for engines). This module must
not import TensorRT, ONNX or Triton at its top: the session imports it."""

from __future__ import annotations

import ctypes
import json
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import onnx

    from .dual_aot import Lowered, Plugin
    from .temporal_session import Part

LAYOUT = "window.cut"
# A cut kernel's output: rows of ROW channels per pixel (HWC8 of 25 or 26 channels)
ROW = 32
# What a piece recomputes from the window's inputs instead of receiving it
VIEW_OPS = frozenset(
    {
        "Cast",
        "Concat",
        "Expand",
        "Gather",
        "Identity",
        "Reshape",
        "Shape",
        "Slice",
        "Squeeze",
        "Unsqueeze",
    }
)


@dataclass
class Piece:
    """One engine of the cut window: its lowered model and the inputs a kernel writes."""

    name: str
    lowered: Lowered
    rows: list[str]


@dataclass
class Cut:
    """The pieces and, per cut, its kernel (op, inputs and output), and the order they
    run in (piece names, and kernels by their outputs)."""

    pieces: list[Piece]
    kernels: list[Plugin]
    order: list[str]


def piece_name(index: int) -> str:
    return f"window_{index}"


def cubin_name(op: str) -> str:
    return f"{op}.cubin"


class _Lowered:
    """Producers and shapes of a lowered model, with its plugins as producers in place of
    their stand-ins."""

    def __init__(self, lowered: Lowered) -> None:
        from .tfix_aot import _Graph

        model = lowered.model
        self.graph = _Graph(model)
        self.lowered = lowered
        self.node_of = {o: n for n in model.graph.node for o in n.output}
        self.plugin_of = {o: p for p in lowered.plugins for o in p.outputs}
        self.inputs = {i.name for i in model.graph.input}
        self.initializers = {i.name for i in model.graph.initializer}
        self.readers = set(lowered.readers)
        # A Shape of a tensor of known dimensions is a constant: no piece needs the tensor
        self.shapes: dict[str, list[int]] = {}
        for node in model.graph.node:
            dims = self.graph.dims.get(node.input[0], []) if node.input else []
            if node.op_type == "Shape" and not node.attribute and dims and all(dims):
                self.shapes[node.output[0]] = dims
        # made of constants alone, and the views, slices and casts of the inputs
        constant: set[str] = set()
        self.cheap: set[str] = set()
        for node in model.graph.node:
            if node.name in self.readers or any(
                o in self.plugin_of for o in node.output
            ):
                continue
            given = [i for i in node.input if i]
            if all(i in self.initializers or i in constant for i in given):
                constant.update(node.output)
                self.cheap.update(node.output)
            elif node.op_type in VIEW_OPS and all(
                i in self.inputs or i in self.initializers or i in self.cheap
                for i in given
            ):
                self.cheap.update(node.output)

    def sources(self, name: str) -> list[str] | None:
        """The tensors `name` is made of (a plugin's inputs for a plugin's output), None
        for an input or an initializer."""
        plugin = self.plugin_of.get(name)
        if plugin is not None:
            return list(plugin.inputs)
        if name in self.shapes:
            return []
        node = self.node_of.get(name)
        return None if node is None else [i for i in node.input if i]

    def closure(
        self, targets: Iterable[str], stop: set[str]
    ) -> tuple[set[str], set[str]]:
        """The tensors made to compute `targets` without crossing `stop`, and the tensors
        of `stop` (and the model's inputs) reached."""
        made: set[str] = set()
        reached: set[str] = set()
        pending = list(targets)
        while pending:
            name = pending.pop()
            if name in made or name in reached:
                continue
            if name in stop or name in self.inputs:
                reached.add(name)
                continue
            sources = self.sources(name)
            if sources is None:
                continue
            made.add(name)
            # a node's other outputs are made with it
            node = self.node_of.get(name)
            if node is not None and name not in self.plugin_of:
                made.update(o for o in node.output if o)
            pending.extend(sources)
        return made, reached

    def users_in(self, name: str, made: set[str]) -> list[onnx.NodeProto]:
        return [
            u
            for u in self.graph.users.get(name, [])
            if any(o in made for o in u.output) and u.name not in self.readers
        ]


def without(lowered: Lowered, outputs: set[str]) -> Lowered:
    """`lowered` with the plugins that make `outputs` left out: their chains, which
    stand in for them, run as TensorRT's own operations."""
    from .dual_aot import Lowered

    plugins = [p for p in lowered.plugins if p.outputs[0] not in outputs]
    return Lowered(lowered.model, plugins, dict(lowered.readers))


def whole(lowered: Lowered) -> Lowered:
    """`lowered` for a window built in one engine: without its DIRECT ops' plugins
    (tfix_aot.register leaves them out)."""
    from .tfix_aot import DIRECT

    direct = {p.outputs[0] for p in lowered.plugins if p.op.startswith(DIRECT)}
    return without(lowered, direct)


def cut(lowered: Lowered) -> Cut | None:
    """The pieces of a lowered window part, cut at its DIRECT ops (None: no such op).
    A cut holds only where each kernel's output is read by the piece right after it
    alone, so that no piece makes a kernel's output: a kernel for which that fails is
    left out of the cut (its chain runs in a piece), and the rest cut again."""
    from .tfix_aot import DIRECT

    kernels = [p for p in lowered.plugins if p.op.startswith(DIRECT)]
    dropped: set[str] = set()
    while kernels:
        kept = without(lowered, dropped)
        planned = _plan(_Lowered(kept), kernels)
        if isinstance(planned, Cut):
            return planned
        dropped.add(planned)
        kernels = [k for k in kernels if k.outputs[0] != planned]
    return None


def _plan(g: _Lowered, kernels: list[Plugin]) -> Cut | str:
    """The cut of `g` at `kernels`, or the output of the first kernel a piece
    other than the one right after it would read (that piece would make it)."""
    outputs = [o.name for o in g.lowered.model.graph.output]
    count = len(kernels) + 1
    direct = {o for k in kernels for o in k.outputs}
    # what each kernel's inputs need, whatever piece makes it
    before = [g.closure(k.inputs, set())[0] for k in kernels]
    targets: list[list[str]] = [[] for _ in range(count)]
    targets[-1] = outputs
    for i, kernel in enumerate(kernels):
        targets[i] = list(kernel.inputs)
    plans: list[tuple[set[str], set[str], list[str]]] = []
    # the tensors later pieces read that an earlier piece makes
    handed: list[str] = []
    for i in reversed(range(count)):
        cut_outputs = set(kernels[i - 1].outputs) if i > 0 else set()
        wanted = list(dict.fromkeys([*targets[i], *handed]))
        stop = (before[i - 1] - g.cheap) | cut_outputs if i > 0 else set()
        made, reached = g.closure(wanted, stop)
        # a received tensor this piece only slices is received as its slices
        for name in sorted(reached - g.inputs - cut_outputs):
            users = g.users_in(name, made)
            if users and all(u.op_type == "Slice" for u in users):
                stop = (stop - {name}) | {u.output[0] for u in users}
        made, reached = g.closure(wanted, stop)
        inside = sorted(made & direct)
        if inside:
            return inside[0]
        own = [t for t in wanted if t not in stop and t not in g.inputs]
        received = sorted(reached - g.inputs - cut_outputs)
        handed = list(dict.fromkeys([*(t for t in wanted if t in stop), *received]))
        plans.insert(0, (made, reached, own))
    if handed:
        # the first piece would receive them: nothing makes them, so no kernel cuts
        return kernels[0].outputs[0]
    pieces, order = [], []
    for i, (made, reached, own) in enumerate(plans):
        if i > 0:
            order.append(kernels[i - 1].outputs[0])
        # a piece with nothing to give (a kernel that reads only the window's inputs, or
        # whose output is the window's) is no engine
        if not own:
            continue
        rows = list(kernels[i - 1].outputs) if i > 0 else []
        pieces.append(Piece(piece_name(i), _piece(g, made, reached, own), rows))
        order.append(piece_name(i))
    return Cut(pieces, kernels, order)


def _piece(
    g: _Lowered, made: set[str], reached: set[str], outputs: list[str]
) -> Lowered:
    """The piece model that makes `made` from `reached` and gives `outputs`, with its
    plugins, their constant stand-ins and their inputs' readers."""
    import numpy as np
    from onnx import helper, numpy_helper

    from .dual_aot import Lowered

    model = g.lowered.model
    plugins = [p for p in g.lowered.plugins if p.outputs[0] in made]
    stood_in = {o for p in plugins for o in p.outputs}
    nodes = [
        helper.make_node(
            "Constant",
            [],
            list(n.output),
            value=numpy_helper.from_array(np.array(g.shapes[n.output[0]], np.int64)),
        )
        if n.output[0] in g.shapes
        else n
        for n in model.graph.node
        if n.name not in g.readers
        and any(o in made for o in n.output)
        and not any(o in stood_in for o in n.output)
    ]
    # the stand-ins depend on nothing, so they lead the graph's order
    shapes, stand_ins = [], []
    for name in sorted(stood_in):
        shape = numpy_helper.from_array(
            np.array(g.graph.dims[name], np.int64), f"{name}/stand_in_shape"
        )
        shapes.append(shape)
        value = helper.make_tensor("value", g.graph.types[name], [1], [0])
        stand_ins.append(
            helper.make_node("ConstantOfShape", [shape.name], [name], value=value)
        )
    nodes = stand_ins + nodes
    used = {i for n in nodes for i in n.input} | {i for p in plugins for i in p.inputs}
    initializers = [t for name, t in g.graph.initializers.items() if name in used]
    initializers += shapes
    readers = {}
    for index, name in enumerate(sorted({i for p in plugins for i in p.inputs})):
        if name in reached:
            continue
        reader = f"tfix/reader_{index}"
        nodes.append(helper.make_node("Identity", [name], [reader], name=reader))
        readers[reader] = name

    def info(name: str) -> onnx.ValueInfoProto:
        # a dimension shape inference left open stays open: the builder fills the
        # piece inputs' in (shape_inputs) from the pieces built before
        dims = [d or None for d in g.graph.dims.get(name, [])]
        return helper.make_tensor_value_info(name, g.graph.types[name], dims or None)

    graph = helper.make_graph(
        nodes,
        "piece",
        [info(n) for n in sorted(reached)],
        [info(n) for n in outputs],
        initializers,
    )
    piece = helper.make_model(graph, opset_imports=list(model.opset_import))
    piece.ir_version = model.ir_version
    return Lowered(piece, plugins, readers)


def row_shapes(kernels: list[Plugin]) -> dict[str, list[int]]:
    """Each cut kernel's output shape, (S, C, H, W), C its op's channels."""
    from .tfix_block_input_kernel import CHANNELS as BLOCK
    from .tfix_conf_input_kernel import CHANNELS as CONF

    shapes = {}
    for kernel in kernels:
        s, h, w = (int(part[1:]) for part in kernel.op.split("_")[2:5])
        channels = CONF.value if kernel.op.startswith("conf_input") else BLOCK.value
        shapes[kernel.outputs[0]] = [s, channels, h, w]
    return shapes


def shape_inputs(model: onnx.ModelProto, shapes: dict[str, list[int]]) -> None:
    """Set the shapes of `model`'s inputs that `shapes` holds (from earlier pieces)."""
    for value in model.graph.input:
        if value.name in shapes:
            dims = value.type.tensor_type.shape.dim
            del dims[:]
            for size in shapes[value.name]:
                dims.add().dim_value = int(size)


def compile_kernels(
    kernels: list[Plugin], sm: int
) -> tuple[list[dict], dict[str, bytes]]:
    """Each cut kernel's launch (op, inputs, output, grid, block, shared memory, function
    name, and the scratch memory it needs: none, which dual_aot.compile_kernel checks)
    and its cubin, compiled for the GPU architecture `sm`."""
    from .dual_aot import MAX_SHARED
    from .tfix_aot import direct_kernel

    launches, cubins = [], {}
    for kernel in kernels:
        compiled, grid = direct_kernel(kernel.op, sm)
        metadata = compiled.metadata
        if int(metadata.shared) > MAX_SHARED:
            raise RuntimeError(
                f"{kernel.op} needs {metadata.shared} B of shared memory"
            )
        cubins[cubin_name(kernel.op)] = bytes(compiled.asm["cubin"])
        launches.append(
            {
                "op": kernel.op,
                "inputs": list(kernel.inputs),
                "output": kernel.outputs[0],
                "grid": list(grid),
                "block": int(metadata.num_warps) * 32,
                "shared": int(metadata.shared),
                "function": str(metadata.name),
                "scratch": int(metadata.global_scratch_size)
                + int(metadata.profile_scratch_size),
            }
        )
    return launches, cubins


def layout(cut_: Cut, launches: list[dict]) -> bytes:
    return json.dumps(
        {
            "pieces": [p.name for p in cut_.pieces],
            "order": cut_.order,
            "kernels": launches,
        },
        indent=1,
    ).encode()


def _check(result: Any, what: str) -> Any:
    from cuda.bindings import driver

    error, *values = result if isinstance(result, tuple) else (result,)
    if error != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Temporal Fix {what} failed: {error}")
    return values[0] if len(values) == 1 else values


class Kernel:
    """A cut kernel loaded on the current device, launched with its buffers' addresses.
    Triton gives a kernel parameters past its buffers (its scratch pointers): they are
    passed as null, which holds only for a kernel that needs no scratch memory."""

    def __init__(self, launch: dict, cubin: bytes) -> None:
        from cuda.bindings import driver

        self.op = str(launch["op"])
        self.inputs = [str(n) for n in launch["inputs"]]
        self.output = str(launch["output"])
        self.grid_x, self.grid_y, self.grid_z = (int(v) for v in launch["grid"])
        self.block = int(launch["block"])
        self.shared = int(launch["shared"])
        if int(launch.get("scratch", -1)) != 0:
            raise RuntimeError(f"Temporal Fix {self.op} needs Triton scratch memory")
        self.module = _check(driver.cuModuleLoadData(cubin), f"{self.op} load")
        try:
            self.function = _check(
                driver.cuModuleGetFunction(
                    self.module, str(launch["function"]).encode()
                ),
                f"{self.op} lookup",
            )
            count = 0
            while (
                driver.cuFuncGetParamInfo(self.function, count)[0]
                == driver.CUresult.CUDA_SUCCESS
            ):
                count += 1
            if count == 0:
                raise RuntimeError(
                    f"The CUDA driver does not give Temporal Fix {self.op}'s"
                    " parameters (cuFuncGetParamInfo needs CUDA 12.4 or later)"
                )
            self.extra = count - len(self.inputs) - 1
            if self.extra < 0:
                raise RuntimeError(
                    f"Temporal Fix {self.op} takes {count} parameters, fewer than"
                    f" its {len(self.inputs) + 1} buffers"
                )
        except BaseException:
            self.close()
            raise

    def launch(self, tensors: dict[str, torch.Tensor], stream: int) -> None:
        from cuda.bindings import driver

        pointers = [tensors[n].data_ptr() for n in (*self.inputs, self.output)]
        pointers += [0] * self.extra
        params = (tuple(pointers), tuple([ctypes.c_void_p] * len(pointers)))
        _check(
            driver.cuLaunchKernel(
                self.function,
                self.grid_x,
                self.grid_y,
                self.grid_z,
                self.block,
                1,
                1,
                self.shared,
                stream,
                params,
                0,
            ),
            f"{self.op} launch",
        )

    def close(self) -> None:
        from cuda.bindings import driver

        _check(driver.cuModuleUnload(self.module), f"{self.op} unload")


def check_layouts(pieces: list[Part], kernels: list[Kernel]) -> None:
    """Every binding a kernel reads is linear and the one it writes is HWC8 (rows of
    32), in every piece that binds it: the layouts the kernels are compiled for."""
    import tensorrt as trt

    for kernel in kernels:
        wanted = dict.fromkeys(kernel.inputs, trt.TensorFormat.LINEAR)
        wanted[kernel.output] = trt.TensorFormat.HWC8
        for piece in pieces:
            for name, form in wanted.items():
                if name not in piece.names:
                    continue
                found = piece.engine.get_tensor_format(name)
                if found != form:
                    raise RuntimeError(
                        f"Temporal Fix {kernel.op}: {piece.name}/{name} is {found},"
                        f" the kernel takes {form}"
                    )


class WindowPieces:
    """The cut window as one engine-like call (temporal_session.Part's call, tensor,
    workspace and use_memory) on the window part's bindings: its pieces and kernels in
    order, the tensors between them in buffers of its own (a kernel's output with room
    for its rows of 32), every call's work in one CUDA graph per set of bound buffers.
    Its kernels are loaded on `device` and unloaded there by close, which a failed
    construction calls before it raises."""

    def __init__(
        self, parts: dict[str, Part], data: dict[str, bytes], device: torch.device
    ) -> None:
        from .temporal_fix_split import window_inputs

        self.device = device
        self.kernels: list[Kernel] = []
        self.graphs: dict[Hashable, torch.cuda.CUDAGraph] = {}
        plan = json.loads(data[LAYOUT])
        self.order_names = [str(name) for name in plan["pieces"]]
        self.pieces = [parts[name] for name in self.order_names]
        self.order = [str(step) for step in plan["order"]]
        self.workspace = max(p.workspace for p in self.pieces)
        try:
            # Loaded in the device's context; the thread's current device is kept
            with torch.cuda.device(device):
                for launch in plan["kernels"]:
                    self.kernels.append(Kernel(launch, data[cubin_name(launch["op"])]))
            check_layouts(self.pieces, self.kernels)
            rows = {k.output for k in self.kernels}
            outer = {*window_inputs(), "output"}
            between: dict[str, torch.Tensor] = {}
            for piece in self.pieces:
                for name in piece.names:
                    if name not in between and name not in outer:
                        between[name] = self._buffer(piece, name, name in rows)
            for kernel in self.kernels:
                if kernel.output not in between:
                    raise RuntimeError(
                        f"Temporal Fix {kernel.op}: no piece reads its output"
                    )
            self.between = between
        except BaseException:
            self.close()
            raise

    def _buffer(self, piece: Part, name: str, rows: bool) -> torch.Tensor:
        shape, dtype = piece.spec(name)
        if not rows:
            return torch.empty(shape, dtype=dtype, device=self.device)
        # Rows of 32 behind the (S, C, H, W) the engine reports: its HWC8 layout
        n, _, h, w = shape
        storage = torch.empty(n * ROW * h * w, dtype=dtype, device=self.device)
        return storage[: n * shape[1] * h * w].view(shape)

    def tensor(self, name: str, device: torch.device) -> torch.Tensor:
        return self.pieces[-1].tensor(name, device)

    def use_memory(self, memory: torch.Tensor) -> None:
        for piece in self.pieces:
            piece.use_memory(memory)

    def call(
        self, key: Hashable, tensors: dict[str, torch.Tensor], stream: torch.cuda.Stream
    ) -> None:
        """The window on `tensors` (the window part's bindings), enqueued on `stream`:
        the graph of `key`, else a run, after which that graph is captured."""
        from .temporal_session import capture_graph

        graph = self.graphs.get(key)
        if graph is not None:
            graph.replay()
            return
        bound = {**tensors, **self.between}
        for piece in self.pieces:
            piece.bind({name: bound[name] for name in piece.names})

        kernels = {k.output: k for k in self.kernels}
        engines = dict(zip(self.order_names, self.pieces, strict=True))

        def enqueue(handle: int) -> bool:
            ran = True
            for step in self.order:
                if step in kernels:
                    kernels[step].launch(bound, handle)
                else:
                    ran = ran and engines[step].enqueue(handle)
            return ran

        if not enqueue(stream.cuda_stream):
            raise RuntimeError("The Temporal Fix window's pieces failed to run.")
        stream.synchronize()
        self.graphs[key] = capture_graph(
            enqueue, self.pieces[0], "Temporal Fix window pieces"
        )

    def close(self) -> None:
        """Frees the graphs, then unloads every kernel on the session's device (the
        thread's current device is kept); the first unload's error is raised after
        every kernel was tried."""
        self.graphs.clear()
        kernels, self.kernels = self.kernels, []
        failure: BaseException | None = None
        with torch.cuda.device(self.device):
            for kernel in kernels:
                try:
                    kernel.close()
                except BaseException as error:
                    failure = failure or error
        if failure is not None:
            raise failure


__all__ = [
    "LAYOUT",
    "Cut",
    "Kernel",
    "Piece",
    "WindowPieces",
    "check_layouts",
    "compile_kernels",
    "cubin_name",
    "cut",
    "layout",
    "piece_name",
    "row_shapes",
    "shape_inputs",
    "whole",
    "without",
]
