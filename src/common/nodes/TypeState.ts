import { EvaluationError, NonNeverType, Type, isSameType } from '@chainner/navi';
import { EdgeData, InputId, IterOutputId, NodeData, OutputId, SchemaId } from '../common-types';
import { log } from '../log';
import { PassthroughInfo, PassthroughMap } from '../PassthroughMap';
import {
    FunctionDefinition,
    FunctionInputAssignmentError,
    FunctionInstance,
} from '../types/function';
import { nullType } from '../types/util';
import { EMPTY_MAP } from '../util';
import { EdgeState, EdgeStateConnection } from './EdgeState';
import type { Edge, Node } from 'reactflow';

const assignmentErrorEquals = (
    a: FunctionInputAssignmentError,
    b: FunctionInputAssignmentError
): boolean => {
    return (
        a.inputId === b.inputId &&
        isSameType(a.assignedType, b.assignedType) &&
        isSameType(a.inputType, b.inputType)
    );
};
const mapEqual = <K, V extends NonNullable<unknown>>(
    a: ReadonlyMap<K, V>,
    b: ReadonlyMap<K, V>,
    eq: (a: V, b: V) => boolean
): boolean => {
    if (a.size !== b.size) return false;
    for (const [key, value] of a) {
        const otherValue = b.get(key);
        if (otherValue === undefined || !eq(value, otherValue)) return false;
    }
    return true;
};
const arrayEqual = <T>(
    a: ReadonlyArray<T>,
    b: ReadonlyArray<T>,
    eq: (a: T, b: T) => boolean
): boolean => {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i += 1) {
        if (!eq(a[i], b[i])) return false;
    }
    return true;
};
const instanceEqual = (a: FunctionInstance, b: FunctionInstance): boolean => {
    if (a.definition !== b.definition) return false;

    if (!mapEqual(a.inputs, b.inputs, isSameType)) return false;
    if (!mapEqual(a.inputSequence, b.inputSequence, isSameType)) return false;
    if (!mapEqual(a.outputs, b.outputs, isSameType)) return false;
    if (!mapEqual(a.outputSequence, b.outputSequence, isSameType)) return false;

    if (!arrayEqual(a.inputErrors, b.inputErrors, assignmentErrorEquals)) return false;

    return true;
};

const connectionEqual = (a: EdgeStateConnection, b: EdgeStateConnection): boolean =>
    a.source === b.source &&
    a.sourceHandle === b.sourceHandle &&
    a.target === b.target &&
    a.targetHandle === b.targetHandle;
const edgeStateEqual = (a: EdgeState, b: EdgeState): boolean =>
    mapEqual(a.byTargetHandle, b.byTargetHandle, connectionEqual) &&
    mapEqual(a.byTarget, b.byTarget, (x, y) => arrayEqual(x, y, connectionEqual));

/** An input that a node's evaluation asked for: the type of its edge, else the node's value. */
interface InputDependency {
    readonly id: InputId;
    readonly edgeType: NonNeverType | undefined;
    readonly value: unknown;
}
/**
 * What a node's function instance was evaluated from. The evaluation is a function of these, so a
 * node whose dependencies are the same has the same instance.
 */
interface Dependencies {
    readonly definition: FunctionDefinition;
    readonly passthrough: PassthroughInfo | undefined;
    // the narrowing maps change in place (useTypeMap): their entries as they were
    readonly outputNarrowing: readonly (readonly [OutputId, Type])[];
    readonly sequenceOutputNarrowing: readonly (readonly [IterOutputId, Type])[];
    /** In the order the evaluation asked for them. */
    readonly inputs: readonly InputDependency[];
}
const entriesOf = <K, V>(map: ReadonlyMap<K, V> | undefined): readonly (readonly [K, V])[] =>
    map ? [...map] : [];
const sameEntries = <K, V>(
    entries: readonly (readonly [K, V])[],
    map: ReadonlyMap<K, V> | undefined
): boolean =>
    entries.length === (map?.size ?? 0) && entries.every(([key, value]) => map?.get(key) === value);

export class TypeState {
    readonly functions: ReadonlyMap<string, FunctionInstance>;

    readonly evaluationErrors: ReadonlyMap<string, EvaluationError>;

    readonly edges: EdgeState;

    private readonly dependencies: ReadonlyMap<string, Dependencies>;

    private constructor(
        functions: TypeState['functions'],
        evaluationErrors: TypeState['evaluationErrors'],
        edges: TypeState['edges'],
        dependencies: TypeState['dependencies']
    ) {
        this.functions = functions;
        this.evaluationErrors = evaluationErrors;
        this.edges = edges;
        this.dependencies = dependencies;
    }

    static readonly empty = new TypeState(EMPTY_MAP, EMPTY_MAP, EdgeState.empty, EMPTY_MAP);

    static create(
        nodesMap: ReadonlyMap<string, Node<NodeData>>,
        rawEdges: readonly Edge<EdgeData>[],
        outputNarrowing: ReadonlyMap<string, ReadonlyMap<OutputId, Type>>,
        sequenceOutputNarrowing: ReadonlyMap<string, ReadonlyMap<IterOutputId, Type>>,
        functionDefinitions: ReadonlyMap<SchemaId, FunctionDefinition>,
        passthrough?: PassthroughMap,
        previousTypeState?: TypeState
    ): TypeState {
        const edges = EdgeState.create(rawEdges);

        const functions = new Map<string, FunctionInstance>();
        const evaluationErrors = new Map<string, EvaluationError>();
        const dependencies = new Map<string, Dependencies>();

        const getSourceType = (id: string, inputId: InputId): NonNeverType | undefined => {
            const edge = edges.get(id, inputId);
            if (edge) {
                const sourceNode = nodesMap.get(edge.source);
                if (sourceNode) {
                    // eslint-disable-next-line @typescript-eslint/no-use-before-define
                    const functionInstance = addNode(sourceNode);
                    return functionInstance?.outputs.get(edge.outputId);
                }
            }
            return undefined;
        };
        const addNode = (n: Node<NodeData>): FunctionInstance | undefined => {
            const cached = functions.get(n.id);
            if (cached) return cached;

            const definition = functionDefinitions.get(n.data.schemaId);
            if (!definition) {
                log.warn(`Unknown schema id ${n.data.schemaId}`);
                return undefined;
            }

            const passthroughInfo = n.data.isPassthrough
                ? passthrough?.get(n.data.schemaId)
                : undefined;
            const narrowing = outputNarrowing.get(n.id);
            const sequenceNarrowing = sequenceOutputNarrowing.get(n.id);

            // Incremental: a node evaluated from the same dependencies as before keeps its
            // instance (and error) without evaluating it again. Asking for its inputs again in
            // the same order adds its sources to the state in the same order as an evaluation.
            // Other node definitions (a backend reload) are other dependencies.
            const previousInstance = previousTypeState?.functions.get(n.id);
            const before = previousTypeState?.dependencies.get(n.id);
            if (
                previousInstance &&
                before &&
                before.definition === definition &&
                before.passthrough === passthroughInfo &&
                sameEntries(before.outputNarrowing, narrowing) &&
                sameEntries(before.sequenceOutputNarrowing, sequenceNarrowing) &&
                before.inputs.every(({ id, edgeType, value }) => {
                    const current = getSourceType(n.id, id);
                    return (
                        current === edgeType &&
                        (current !== undefined || n.data.inputData[id] === value)
                    );
                })
            ) {
                const error = previousTypeState?.evaluationErrors.get(n.id);
                if (error) evaluationErrors.set(n.id, error);
                dependencies.set(n.id, before);
                functions.set(n.id, previousInstance);
                return previousInstance;
            }

            const inputs: InputDependency[] = [];
            let instance;
            try {
                instance = FunctionInstance.fromPartialInputs(
                    definition,
                    (id): NonNeverType | undefined => {
                        const edgeSource = getSourceType(n.id, id);
                        inputs.push({
                            id,
                            edgeType: edgeSource,
                            value: edgeSource ? undefined : n.data.inputData[id],
                        });
                        if (edgeSource) {
                            return edgeSource;
                        }

                        const inputValue = n.data.inputData[id];

                        if (inputValue !== undefined) {
                            const foo = definition.inputDataAdapters.get(id)?.(inputValue);
                            if (foo !== undefined) {
                                return foo;
                            }
                        }

                        if (inputValue === undefined && definition.inputNullable.has(id)) {
                            return nullType;
                        }

                        return undefined;
                    },
                    narrowing,
                    sequenceNarrowing,
                    passthroughInfo
                );
            } catch (error) {
                if (error instanceof EvaluationError) {
                    evaluationErrors.set(n.id, error);
                } else {
                    throw error;
                }
                instance = definition.defaultInstance;
            }

            if (previousInstance && instanceEqual(previousInstance, instance)) {
                instance = previousInstance;
            }

            dependencies.set(n.id, {
                definition,
                passthrough: passthroughInfo,
                outputNarrowing: entriesOf(narrowing),
                sequenceOutputNarrowing: entriesOf(sequenceNarrowing),
                inputs,
            });
            functions.set(n.id, instance);
            return instance;
        };

        for (const n of nodesMap.values()) {
            addNode(n);
        }

        // Nothing the types depend on changed (e.g. a node only moved): the previous state,
        // so that everything that depends on it keeps its work.
        if (
            previousTypeState &&
            mapEqual(functions, previousTypeState.functions, (a, b) => a === b) &&
            mapEqual(
                evaluationErrors,
                previousTypeState.evaluationErrors,
                (a, b) => a.message === b.message
            ) &&
            edgeStateEqual(edges, previousTypeState.edges)
        ) {
            return previousTypeState;
        }

        return new TypeState(functions, evaluationErrors, edges, dependencies);
    }
}
