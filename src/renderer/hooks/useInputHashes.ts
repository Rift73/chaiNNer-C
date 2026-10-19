import { MutableRefObject, useEffect, useRef, useState } from 'react';
import { Edge, Node, useReactFlow } from 'reactflow';
import { EdgeData, NodeData } from '../../common/common-types';
import { SchemaMap } from '../../common/SchemaMap';
import { EMPTY_MAP, deriveUniqueId, stringifyTargetHandle } from '../../common/util';

/**
 * Given a graph, this will returns a map from node id to the input hash of the node with that id.
 *
 * An input hash is a representation of the inputs for a node. If the input hash changes, at least
 * of its inputs changed. At least, that's the idea. Due to the imperfect nature of hashes, input
 * hashes only guarantee that if the input hash doesn't change, that the inputs for a node did not
 * change.
 *
 * Each node's hash is one hash of everything its inputs are: its schema, whether it passes its
 * inputs through, and per input the value (`null` for none) or, for a connected input, the hash of
 * the connected node in a list (which no value is).
 */
export const computeInputHashes = (
    nodes: readonly Node<NodeData>[],
    edges: readonly Edge<EdgeData>[],
    schemata: SchemaMap
): Map<string, string> => {
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const byTargetHandle = new Map(
        edges.filter((e) => e.targetHandle).map((e) => [e.targetHandle!, e])
    );

    const hashes = new Map<string, string>();
    const getInputHash = (node: Node<NodeData>): string => {
        let hash = hashes.get(node.id);
        if (hash) return hash;

        const schema = schemata.get(node.data.schemaId);
        const inputs: (string | readonly [string] | null)[] = [
            node.data.schemaId,
            String(node.data.isPassthrough ?? false),
        ];
        for (const input of schema.inputs) {
            const connectedEdge = byTargetHandle.get(
                stringifyTargetHandle({ nodeId: node.id, inputId: input.id })
            );
            if (connectedEdge) {
                const source = byId.get(connectedEdge.source);
                if (source) {
                    // input gets passed by the connected node
                    inputs.push([getInputHash(source)]);
                    // eslint-disable-next-line no-continue
                    continue;
                }
            }

            const value = node.data.inputData[input.id];
            // eslint-disable-next-line eqeqeq
            inputs.push(value == undefined ? null : String(value));
        }

        hash = deriveUniqueId(JSON.stringify(inputs));
        hashes.set(node.id, hash);
        return hash;
    };

    for (const node of nodes) {
        getInputHash(node);
    }

    return hashes;
};

const sameHashes = (a: ReadonlyMap<string, string>, b: ReadonlyMap<string, string>): boolean => {
    if (a.size !== b.size) return false;
    for (const [id, hash] of a) {
        if (b.get(id) !== hash) return false;
    }
    return true;
};

/**
 * The input hashes of the chain, as state (a change re-renders, so what shows them is never a
 * render behind) and as a ref (the newest, for reads outside of rendering). They are computed
 * after the chain changed, once React Flow has the new nodes and edges; a change that leaves
 * every hash as it is (e.g. a node moved) keeps both.
 */
export const useInputHashes = (
    schemata: SchemaMap,
    deps: readonly unknown[]
): readonly [ReadonlyMap<string, string>, MutableRefObject<ReadonlyMap<string, string>>] => {
    const ref = useRef<ReadonlyMap<string, string>>(EMPTY_MAP);
    const [hashes, setHashes] = useState<ReadonlyMap<string, string>>(EMPTY_MAP);

    const { getNodes, getEdges } = useReactFlow();
    useEffect(() => {
        const newHashes = computeInputHashes(getNodes(), getEdges(), schemata);
        if (!sameHashes(ref.current, newHashes)) {
            ref.current = newHashes;
            setHashes(newHashes);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, deps);

    return [hashes, ref];
};
