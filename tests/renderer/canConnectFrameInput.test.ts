import { Edge, Node } from 'reactflow';
import { expect, test } from 'vitest';
import {
    CategoryId,
    EdgeData,
    Input,
    InputId,
    InputValue,
    NodeData,
    NodeGroupId,
    NodeSchema,
    Output,
    OutputId,
    SchemaId,
} from '../../src/common/common-types';
import { checkNodeValidity } from '../../src/common/nodes/checkNodeValidity';
import { getFirstPossibleInput } from '../../src/common/nodes/connectedInputs';
import { ChainLineage } from '../../src/common/nodes/lineage';
import { parseFunctionDefinitions } from '../../src/common/nodes/parseFunctionDefinitions';
import { TypeState } from '../../src/common/nodes/TypeState';
import { SchemaMap } from '../../src/common/SchemaMap';
import {
    EMPTY_MAP,
    parseTargetHandle,
    stringifySourceHandle,
    stringifyTargetHandle,
} from '../../src/common/util';
import { canConnect } from '../../src/renderer/helpers/canConnect';

// A VapourSynth family node: Method picks whether Clip also takes an image, Other follows Clip's
// kind, and the output is what Clip got.
const MAIN = 0 as InputId;
const METHOD = 1 as InputId;
const OTHER = 2 as InputId;
const PLAIN = 3 as InputId;
const NAMES_OTHER = 4 as InputId;
const IMAGE_ONLY = 5 as InputId;
const OUT = 0 as OutputId;

const socket = (id: InputId, label: string, type: string, followsInputs = false): Input => ({
    id,
    type: [type, 'null'],
    conversions: [],
    kind: 'generic',
    label,
    optional: true,
    hasHandle: true,
    hint: false,
    suggest: false,
    followsInputs,
});
const output = (type: string): Output => ({
    id: OUT,
    type,
    label: 'Out',
    kind: 'generic',
    hasHandle: true,
    suggest: false,
});
const nodeSchema = (schemaId: string, inputs: Input[], outputs: Output[]): NodeSchema => ({
    name: schemaId,
    category: 'test' as CategoryId,
    nodeGroup: 'test' as NodeGroupId,
    description: '',
    seeAlso: [],
    icon: '',
    kind: 'regularNode',
    inputs,
    outputs,
    groupLayout: inputs.map((i) => i.id),
    iteratorInputs: [],
    iteratorOutputs: [],
    suggestions: [],
    schemaId: schemaId as SchemaId,
    hasSideEffects: false,
    deprecated: false,
    features: [],
});

const method = {
    ...socket(METHOD, 'Method', 'string'),
    kind: 'dropdown',
    optional: false,
    hasHandle: false,
    options: [
        { option: 'Blur', value: 'blur', type: '"frame"' },
        { option: 'Degrain', value: 'degrain', type: '"clip"' },
    ],
    def: 'blur',
    preferredStyle: 'dropdown',
    labelStyle: 'default',
    groups: [],
} as Input;

const schemata = new SchemaMap([
    nodeSchema('image', [], [output('Image')]),
    nodeSchema('clip', [], [output('VideoClip')]),
    nodeSchema('audio', [], [output('Audio')]),
    nodeSchema(
        'family',
        [
            socket(
                MAIN,
                'Clip',
                'match Input1 { "frame" => VideoClip | Image, _ => VideoClip }',
                true
            ),
            method,
            socket(OTHER, 'Other', 'match Input0 { Image => Image, _ => VideoClip }', true),
            socket(PLAIN, 'Plain', 'Image | Audio'),
            // As Dither's palette: names another input, but connects whatever that holds and
            // shows the error after, as every node does but those that take images for clips.
            socket(NAMES_OTHER, 'Names other', 'match Input3 { Image => Image, _ => Audio }'),
            socket(IMAGE_ONLY, 'Image only', 'Image'),
        ],
        [output('match Input0 { Image => Image, _ => VideoClip }')]
    ),
]);
const functionDefinitions = parseFunctionDefinitions(schemata.schemata);

// node IDs are UUIDs: handles keep the node ID as their first 36 characters
const id = (i: number) => `00000000-0000-4000-8000-${i.toString(16).padStart(12, '0')}`;
const IMAGE = 0;
const CLIP = 1;
const AUDIO = 2;
const FAMILY = 3;

const edge = (source: number, input: InputId): Edge<EdgeData> => ({
    id: `${source}-${input}`,
    source: id(source),
    sourceHandle: stringifySourceHandle({ nodeId: id(source), outputId: OUT }),
    target: id(FAMILY),
    targetHandle: stringifyTargetHandle({ nodeId: id(FAMILY), inputId: input }),
});

const state = (picked: InputValue, edges: Edge<EdgeData>[]) => {
    const node = (i: number, schemaId: string, inputData: NodeData['inputData']) => ({
        id: id(i),
        position: { x: 0, y: 0 },
        data: { id: id(i), schemaId: schemaId as SchemaId, inputData },
    });
    const nodes: Node<NodeData>[] = [
        node(IMAGE, 'image', {}),
        node(CLIP, 'clip', {}),
        node(AUDIO, 'audio', {}),
        node(FAMILY, 'family', { [METHOD]: picked }),
    ];
    const types = TypeState.create(
        new Map(nodes.map((n) => [n.id, n])),
        edges,
        EMPTY_MAP,
        EMPTY_MAP,
        functionDefinitions
    );
    const lineage = ChainLineage.create(schemata, nodes, edges);
    return {
        family: types.functions.get(id(FAMILY))!,
        imageType: types.functions.get(id(IMAGE))!.outputs.get(OUT)!,
        connect: (source: number, input: InputId) =>
            canConnect(
                { nodeId: id(source), outputId: OUT },
                { nodeId: id(FAMILY), inputId: input },
                types,
                lineage
            ),
    };
};

test('Clip takes an image only under a method that looks at one frame', () => {
    expect(state('blur', []).connect(IMAGE, MAIN).isValid).toBe(true);
    expect(state('blur', []).connect(CLIP, MAIN).isValid).toBe(true);
    expect(state('degrain', []).connect(CLIP, MAIN).isValid).toBe(true);

    const refused = state('degrain', []).connect(IMAGE, MAIN);
    if (refused.isValid) throw new Error('An image connected to a temporal method.');
    expect(refused.reason).toBe(
        'This input takes a clip, not single images: the node works on the clip as a whole or looks at other frames. Open the video with Load Clip, or turn images into a clip with Frames to Clip.'
    );
});

test('a refused wire between clips and images says which node converts them', () => {
    const { connect } = state('blur', []);
    expect(connect(CLIP, IMAGE_ONLY)).toEqual({
        isValid: false,
        reason: 'This input takes an image. Add Clip Frames to turn the clip into images.',
    });
    // an input that takes images among other things too
    expect(connect(CLIP, PLAIN)).toEqual({
        isValid: false,
        reason: 'This input takes an image. Add Clip Frames to turn the clip into images.',
    });
    // a mismatch without a clip keeps the general text
    const audio = connect(AUDIO, IMAGE_ONLY);
    if (audio.isValid) throw new Error('Audio connected to an image input.');
    expect(audio.reason).toMatch(
        /^Input Image only cannot be connected with an incompatible value\./
    );
});

test('a connection can replace one of the other kind on the same input', () => {
    // the input is narrowed to what it holds; what would replace it is judged by what it accepts
    expect(state('blur', [edge(CLIP, MAIN)]).connect(IMAGE, MAIN).isValid).toBe(true);
    expect(state('blur', [edge(IMAGE, MAIN)]).connect(CLIP, MAIN).isValid).toBe(true);
    expect(state('blur', [edge(IMAGE, PLAIN)]).connect(AUDIO, PLAIN).isValid).toBe(true);
    expect(state('blur', [edge(AUDIO, PLAIN)]).connect(IMAGE, PLAIN).isValid).toBe(true);
    expect(state('blur', [edge(AUDIO, PLAIN)]).connect(CLIP, PLAIN).isValid).toBe(false);
});

test('a second clip input follows what the main one holds', () => {
    expect(state('blur', [edge(IMAGE, MAIN)]).connect(IMAGE, OTHER).isValid).toBe(true);
    expect(state('blur', [edge(IMAGE, MAIN)]).connect(CLIP, OTHER).isValid).toBe(false);
    expect(state('blur', [edge(CLIP, MAIN)]).connect(CLIP, OTHER).isValid).toBe(true);
    expect(state('blur', [edge(CLIP, MAIN)]).connect(IMAGE, OTHER).isValid).toBe(false);
});

test('the output is an image when an image goes in', () => {
    const type = (edges: Edge<EdgeData>[]) => state('blur', edges).family.outputs.get(OUT)!;
    expect(type([edge(IMAGE, MAIN)]).toString()).toContain('Image');
    expect(type([edge(CLIP, MAIN)]).toString()).toContain('VideoClip');
    expect(type([edge(CLIP, MAIN)]).toString()).not.toContain('Image');
});

test('picking a temporal method with an image connected is an input error', () => {
    expect(state('blur', [edge(IMAGE, MAIN)]).family.inputErrors).toEqual([]);
    const { inputErrors } = state('degrain', [edge(IMAGE, MAIN)]).family;
    expect(inputErrors.map((e) => e.inputId)).toEqual([MAIN]);
});

test('an input the backend did not mark is judged by its default, as before', () => {
    // Plain holds audio, so this input's type is Audio; an image still connects, and only then
    // does the node show the mismatch.
    const { connect, family } = state('blur', [edge(AUDIO, PLAIN)]);
    expect(family.inputExpectations.get(NAMES_OTHER)?.toString()).toContain('Image');
    expect(connect(IMAGE, NAMES_OTHER).isValid).toBe(true);
    expect(connect(CLIP, NAMES_OTHER).isValid).toBe(false);
});

test('an edge dropped onto a node goes by what the node accepts now', () => {
    const image = state('blur', []).imageType;
    const blur = state('blur', []).family;
    expect(getFirstPossibleInput(blur.definition, image, false, blur)).toBe(MAIN);
    // Under a temporal method Clip refuses it: the next input that takes it.
    const degrain = state('degrain', []).family;
    expect(getFirstPossibleInput(degrain.definition, image, false, degrain)).toBe(PLAIN);
    // A node not yet made has only its definition.
    expect(getFirstPossibleInput(degrain.definition, image, false)).toBe(MAIN);
});

test("the node's error mark names the input and the node that converts clips and images", () => {
    const validity = (picked: InputValue, edges: Edge<EdgeData>[]) =>
        checkNodeValidity({
            schema: schemata.get('family' as SchemaId),
            inputData: { [METHOD]: picked },
            connectedInputs: new Set(edges.map((e) => parseTargetHandle(e.targetHandle!).inputId)),
            functionInstance: state(picked, edges).family,
            chainLineage: undefined,
            nodeId: id(FAMILY),
        });

    // an image connected, then a method that takes only clips
    expect(validity('degrain', [edge(IMAGE, MAIN)])).toEqual({
        isValid: false,
        reason: 'Clip takes a clip, not single images: the node works on the clip as a whole or looks at other frames. Open the video with Load Clip, or turn images into a clip with Frames to Clip.',
    });
    expect(validity('blur', [edge(CLIP, IMAGE_ONLY)])).toEqual({
        isValid: false,
        reason: 'Image only takes an image. Add Clip Frames to turn the clip into images.',
    });
    expect(validity('blur', [edge(CLIP, PLAIN)])).toEqual({
        isValid: false,
        reason: 'Plain takes an image. Add Clip Frames to turn the clip into images.',
    });
    // every other mismatch keeps the trace
    const other = validity('blur', [edge(AUDIO, IMAGE_ONLY)]);
    if (other.isValid) throw new Error('Audio connected to an image input.');
    expect(other.reason).toMatch(/^Input Image only was connected with an incompatible value\./);
});
