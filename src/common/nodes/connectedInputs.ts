import { Type } from '@chainner/navi';
import { EdgeData, InputId, OutputId } from '../common-types';
import { FunctionDefinition } from '../types/function';
import { EMPTY_SET, parseTargetHandle } from '../util';
import { isAutoIterable } from './lineage';
import type { Edge } from 'reactflow';

/**
 * The connected inputs of every node, by the edges they were read from. Every node's validity is
 * checked against the same edges after each edge change: one pass over them serves all nodes.
 * Edge lists are replaced on every change, never changed in place.
 */
const connectedInputsByEdges = new WeakMap<
    readonly Edge<EdgeData>[],
    ReadonlyMap<string, ReadonlySet<InputId>>
>();

export const getConnectedInputs = (
    nodeId: string,
    edges: readonly Edge<EdgeData>[]
): ReadonlySet<InputId> => {
    let byNode = connectedInputsByEdges.get(edges);
    if (byNode === undefined) {
        const map = new Map<string, Set<InputId>>();
        for (const e of edges) {
            if (e.targetHandle) {
                let inputs = map.get(e.target);
                if (inputs === undefined) {
                    inputs = new Set();
                    map.set(e.target, inputs);
                }
                inputs.add(parseTargetHandle(e.targetHandle).inputId);
            }
        }
        byNode = map;
        connectedInputsByEdges.set(edges, byNode);
    }
    return byNode.get(nodeId) ?? EMPTY_SET;
};

/**
 * Returns the first compatible input that can be assigned the given type.
 * If `isIterated` is true, then the input will be given an iterator.
 */
export const getFirstPossibleInput = (
    fn: FunctionDefinition,
    type: Type,
    isIterated: boolean
): InputId | undefined => {
    return fn.schema.inputs.find((i) => {
        // check if the input has a handle
        if (!i.hasHandle) return false;

        if (!isAutoIterable(fn.schema)) {
            const inputIterated = fn.schema.iteratorInputs.some((info) =>
                info.inputs.includes(i.id)
            );
            if (inputIterated !== isIterated) return false;
        }

        return fn.canAssignInput(i.id, type);
    })?.id;
};
export const getFirstPossibleOutput = (
    outputFn: FunctionDefinition,
    inputFn: FunctionDefinition,
    inputId: InputId
): OutputId | undefined => {
    if (!inputFn.hasInput(inputId)) return undefined;

    let allowIterated = false;
    let allowNonIterated = false;
    if (isAutoIterable(inputFn.schema)) {
        allowIterated = true;
        allowNonIterated = true;
    } else {
        const inputIterated = inputFn.schema.iteratorInputs.some((i) => i.inputs.includes(inputId));
        allowIterated = inputIterated;
        allowNonIterated = !inputIterated;
    }

    return outputFn.schema.outputs.find((o) => {
        // check if the output has a handle
        if (!o.hasHandle) return false;

        // check iterator compatibility
        const outputIterated = outputFn.schema.iteratorOutputs.some((info) =>
            info.outputs.includes(o.id)
        );
        if (!allowIterated && outputIterated) return false;
        if (!allowNonIterated && !isAutoIterable(outputFn.schema) && !outputIterated) return false;

        // type check
        const type = outputFn.outputDefaults.get(o.id);
        return type && inputFn.canAssignInput(inputId, type);
    })?.id;
};
