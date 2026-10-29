from __future__ import annotations

from typing import Literal, TypedDict, Union

from api import NodeId

from .chain import (
    Chain,
    CollectorNode,
    Edge,
    EdgeSource,
    EdgeTarget,
    FunctionNode,
    GeneratorNode,
)


class JsonEdgeInput(TypedDict):
    type: Literal["edge"]
    id: NodeId
    index: int


class JsonValueInput(TypedDict):
    type: Literal["value"]
    value: object


JsonInput = Union[JsonEdgeInput, JsonValueInput]


class JsonNode(TypedDict):
    id: NodeId
    schemaId: str
    inputs: list[JsonInput]
    parent: NodeId | None
    nodeType: str


MAX_CHAIN_DEPTH = 500
"""The most nodes in a row (the longest series of connected nodes) a run takes. Both
executors recurse once per node in a row (the native engine's Chain walks and its driver
thread's stack; process.Executor's demand-driven node calls), so a longer chain would end
in a recursion error, or a crash on the engine's stack, instead of this one's message. A
connection into a lazy input counts as LAZY_LINK_DEPTH nodes."""

LAZY_LINK_DEPTH = 5
"""The nodes in a row a connection into a lazy input (Conditional, Logic Operation)
counts as: reading it runs its upstream inside the reading node's run, so on the native
engine's driver each such link nests about 8 Python frames where a node in a row nests
2. The engine runs at most 124 such links in a row; 99 is the most a chain may have."""


class IndexEdge:
    """A connection as the chain file gives it: parse_json checks from_index (any JSON
    value) against the source node's outputs."""

    def __init__(
        self, from_id: NodeId, from_index: object, to_id: NodeId, to_index: int
    ) -> None:
        self.from_id = from_id
        self.from_index = from_index
        self.to_id = to_id
        self.to_index = to_index


def parse_json(json: list[JsonNode]) -> Chain:
    """The chain of a /run request. A chain file is untrusted (people share them): a
    connection or input the node does not have, or a cycle, is a ValueError naming it, as
    is a series of more than MAX_CHAIN_DEPTH connected nodes."""
    chain = Chain()

    index_edges: list[IndexEdge] = []

    for json_node in json:
        if json_node["nodeType"] == "generator":
            node = GeneratorNode(json_node["id"], json_node["schemaId"])
        elif json_node["nodeType"] == "collector":
            node = CollectorNode(json_node["id"], json_node["schemaId"])
        else:
            node = FunctionNode(json_node["id"], json_node["schemaId"])
        chain.add_node(node)

        inputs = node.data.inputs
        if len(json_node["inputs"]) > len(inputs):
            raise ValueError(
                f"The node {node.id} ({node.data.name}) has {len(inputs)} inputs,"
                f" not {len(json_node['inputs'])}."
            )
        for index, i in enumerate(json_node["inputs"]):
            if i["type"] == "edge":
                index_edges.append(IndexEdge(i["id"], i["index"], node.id, index))
            else:
                chain.inputs.set(node.id, inputs[index].id, i["value"])

    for index_edge in index_edges:
        target_node = chain.nodes[index_edge.to_id].data
        source = chain.nodes.get(index_edge.from_id)
        outputs = [] if source is None else source.data.outputs
        if not (
            isinstance(index_edge.from_index, int)
            and 0 <= index_edge.from_index < len(outputs)
        ):
            raise ValueError(
                f"The input {target_node.inputs[index_edge.to_index].label} of the node"
                f" {index_edge.to_id} ({target_node.name}) is connected to output"
                f" {index_edge.from_index!r} of the node {index_edge.from_id!r}, which"
                " the chain does not have."
            )
        source_node = chain.nodes[index_edge.from_id].data

        chain.add_edge(
            Edge(
                EdgeSource(
                    index_edge.from_id,
                    source_node.outputs[index_edge.from_index].id,
                ),
                EdgeTarget(
                    index_edge.to_id,
                    target_node.inputs[index_edge.to_index].id,
                ),
            )
        )

    # The editor makes no cycle; a crafted one would recurse without end in the run.
    # The same pass measures the longest series of connected nodes: the nodes in a row
    # up to each node (a lazy input's connection weighing LAZY_LINK_DEPTH), and the
    # first of them.
    waiting = {node_id: len(chain.edges_to(node_id)) for node_id in chain.nodes}
    lazy = {
        node_id: {i.id for i in node.data.inputs if i.lazy}
        for node_id, node in chain.nodes.items()
    }
    in_a_row = dict.fromkeys(chain.nodes, 1)
    first = {node_id: node_id for node_id in chain.nodes}
    ready = [node_id for node_id, count in waiting.items() if count == 0]
    while ready:
        source = ready.pop()
        for edge in chain.edges_from(source):
            target = edge.target.id
            weight = LAZY_LINK_DEPTH if edge.target.input_id in lazy[target] else 1
            if in_a_row[source] + weight > in_a_row[target]:
                in_a_row[target] = in_a_row[source] + weight
                first[target] = first[source]
            waiting[target] -= 1
            if waiting[target] == 0:
                ready.append(target)
    stuck = sorted(node_id for node_id, count in waiting.items() if count > 0)
    if stuck:
        raise ValueError(
            "The chain's connections form a cycle, so these nodes cannot run:"
            f" {', '.join(stuck[:8])}{', ...' if len(stuck) > 8 else ''}."
        )
    last = max(chain.nodes, key=in_a_row.__getitem__, default=None)
    if last is not None and in_a_row[last] > MAX_CHAIN_DEPTH:
        raise ValueError(
            f"This chain has {in_a_row[last]} nodes in a row (from"
            f" {chain.nodes[first[last]].data.name} to {chain.nodes[last].data.name});"
            f" a run supports at most {MAX_CHAIN_DEPTH}. Split the chain, saving"
            " intermediate images."
        )

    return chain
