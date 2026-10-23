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


class IndexEdge:
    def __init__(
        self, from_id: NodeId, from_index: int, to_id: NodeId, to_index: int
    ) -> None:
        self.from_id = from_id
        self.from_index = from_index
        self.to_id = to_id
        self.to_index = to_index


def parse_json(json: list[JsonNode]) -> Chain:
    """The chain of a /run request. A chain file is untrusted (people share them): a
    connection or input the node does not have, or a cycle, is a ValueError naming it."""
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
    waiting = {node_id: len(chain.edges_to(node_id)) for node_id in chain.nodes}
    ready = [node_id for node_id, count in waiting.items() if count == 0]
    while ready:
        for edge in chain.edges_from(ready.pop()):
            waiting[edge.target.id] -= 1
            if waiting[edge.target.id] == 0:
                ready.append(edge.target.id)
    stuck = sorted(node_id for node_id, count in waiting.items() if count > 0)
    if stuck:
        raise ValueError(
            "The chain's connections form a cycle, so these nodes cannot run:"
            f" {', '.join(stuck[:8])}{', ...' if len(stuck) > 8 else ''}."
        )

    return chain
