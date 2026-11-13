import { SourceDocument, evaluate, parseExpression } from '@chainner/navi';
import { readFileSync } from 'fs';
import path from 'path';
import { Node } from 'reactflow';
import { expect, test, vi } from 'vitest';
import {
    CategoryId,
    InputId,
    NodeData,
    NodeGroupId,
    NodeSchema,
    OutputId,
    SchemaId,
} from '../../src/common/common-types';
import { parseFunctionDefinitions } from '../../src/common/nodes/parseFunctionDefinitions';
import { TypeState } from '../../src/common/nodes/TypeState';
import { getChainnerScope } from '../../src/common/types/chainner-scope';

const { engine } = vi.hoisted(() => ({ engine: { loaded: false } }));

// the app loads the regex WebAssembly in the background; here it loads at once and is reported
// as loaded only when the test says so
vi.mock('../../src/common/rust-regex', async () => {
    const web = await import('rregex/lib/web');
    web.initSync(readFileSync(path.resolve('node_modules/rregex/lib/rregex.wasm')));
    return { RRegex: web.RRegex, isRegexLoaded: () => engine.loaded };
});

const type = (code: string) =>
    evaluate(parseExpression(new SourceDocument(code, 'test')), getChainnerScope()).toString();

const REPLACE = 'test:regex:replace' as SchemaId;
const schema: NodeSchema = {
    name: 'Replace',
    category: 'test' as CategoryId,
    nodeGroup: 'test' as NodeGroupId,
    description: '',
    seeAlso: [],
    icon: '',
    kind: 'regularNode',
    inputs: [
        {
            id: 0 as InputId,
            type: 'string',
            conversions: [],
            kind: 'text',
            label: 'Text',
            optional: false,
            hasHandle: true,
            hint: false,
            suggest: false,
            multiline: false,
        } as never,
    ],
    outputs: [
        {
            id: 0 as OutputId,
            type: 'regexReplace(Input0, "a", "b", inf)',
            label: 'Text',
            kind: 'generic',
            hasHandle: true,
            suggest: false,
        },
    ],
    groupLayout: [0 as InputId],
    iteratorInputs: [],
    iteratorOutputs: [],
    suggestions: [],
    schemaId: REPLACE,
    hasSideEffects: false,
    deprecated: false,
    features: [],
};

test('before the regex engine has loaded, a regex is not reported as invalid', () => {
    engine.loaded = false;
    expect(type('regexReplace("aaa", "a", "b", 2)')).toBe('string | Error { message: string }');
    expect(type('regexFind("abc", "b(c)", "{1}")')).toBe('string | Error { message: string }');

    engine.loaded = true;
    expect(type('regexReplace("aaa", "a", "b", 2)')).toBe('"bba"');
    expect(type('regexFind("abc", "b(c)", "{1}")')).toBe('"c"');
    expect(type('regexReplace("aaa", "(", "b", 2)')).toMatch(/^Error \{ message: "Invalid regex/);
});

test("a node's types are evaluated again once the regex engine has loaded", () => {
    const definitions = parseFunctionDefinitions([schema]);
    const node: Node<NodeData> = {
        id: 'node',
        position: { x: 0, y: 0 },
        data: { id: 'node', schemaId: REPLACE, inputData: { [0 as InputId]: 'aaa' } },
    };
    const create = (previous?: TypeState) =>
        TypeState.create(
            new Map([[node.id, node]]),
            [],
            new Map(),
            new Map(),
            definitions,
            undefined,
            previous
        );
    const output = (state: TypeState) =>
        state.functions
            .get('node')
            ?.outputs.get(0 as OutputId)
            ?.toString();

    engine.loaded = false;
    const before = create();
    expect(output(before)).toBe('string');

    engine.loaded = true;
    expect(output(create(before))).toBe('"bbb"');
});

// as the backend reads a pattern: a name is letters and digits of any script, and `_`
test.each(['名前', 'über', '１', 'a_b'])('the replacement name {%s} is a name', (name) => {
    engine.loaded = true;
    expect(type(`formatPattern("{${name}}", "x")`)).toBe(
        `Error { message: "Invalid reference {${name}} in pattern." }`
    );
});
test.each(['名前', 'über', 'a_b'])('a regex group named %s is a replacement', (name) => {
    engine.loaded = true;
    expect(type(`regexFind("ab", "(?P<${name}>b)", "[{${name}}]")`)).toBe('"[b]"');
});
test.each(['a-b', 'a b', 'a.b'])('the replacement name {%s} is not a name', (name) => {
    expect(type(`formatPattern("{${name}}", "x")`)).toMatch(/is not a valid replacement/);
});
