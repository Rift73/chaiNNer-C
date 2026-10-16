import { readFileSync } from 'fs';
import path from 'path';
import { initSync } from 'rregex/lib/web';
import { expect, test } from 'vitest';
import { CategoryMap } from '../../src/common/CategoryMap';
import { CategoryId, NodeGroupId, NodeSchema, SchemaId } from '../../src/common/common-types';
import { getBestMatch, getMatchingNodes } from '../../src/renderer/helpers/nodeSearchFuncs';

// The renderer fetches the regex engine's wasm by URL, which Node can't do; load the file directly.
initSync(readFileSync(path.resolve('node_modules/rregex/lib/rregex.wasm')));

const schema = (name: string, category: string, keywords?: readonly string[]): NodeSchema => ({
    name,
    category: category as CategoryId,
    nodeGroup: `${category}/group` as NodeGroupId,
    description: '',
    seeAlso: [],
    icon: 'BsQuestion',
    kind: 'regularNode',
    inputs: [],
    outputs: [],
    groupLayout: [],
    iteratorInputs: [],
    iteratorOutputs: [],
    keyInfo: null,
    suggestions: [],
    schemaId: `chainner:${category}:${name.toLowerCase().replace(/ /g, '_')}` as SchemaId,
    hasSideEffects: false,
    deprecated: false,
    features: [],
    ...(keywords ? { keywords } : {}),
});

const denoise = schema('Denoise', 'vapoursynth', ['BM3D', 'vsdenoise.bm3d', 'NL-Means']);
const sharpen = schema('Sharpen and Blur', 'vapoursynth', ['CAS', 'Blur', 'vs.cas']);
const blur = schema('Blur', 'image_filter');
const cascade = schema('Cascade', 'image_filter');
const resize = schema('Resize', 'image_dimension');
const schemata = [denoise, sharpen, blur, cascade, resize];
const categories = CategoryMap.EMPTY;

test('matches keyword "bm3d" to Denoise', () => {
    const matching = getMatchingNodes('bm3d', schemata, categories);
    expect(matching).toStrictEqual([denoise]);
    expect(getBestMatch('bm3d', matching, categories)).toBe(denoise);

    // the same acronym-style matching as names: "nlm" finds the keyword "NL-Means"
    expect(getMatchingNodes('nlm', schemata, categories)).toStrictEqual([denoise]);
});

test('schemata without keywords still match by name only', () => {
    expect(getMatchingNodes('resize', schemata, categories)).toStrictEqual([resize]);
    expect(getMatchingNodes('', schemata, categories)).toBe(schemata);
});

test('best match prefers an exact keyword over a partial name', () => {
    const matching = getMatchingNodes('cas', schemata, categories);
    expect(matching).toContain(sharpen);
    expect(matching).toContain(cascade);
    expect(getBestMatch('cas', matching, categories)).toBe(sharpen);
});

test('best match prefers a name over an equally good keyword', () => {
    const matching = getMatchingNodes('blur', schemata, categories);
    expect(matching).toStrictEqual([sharpen, blur]);
    expect(getBestMatch('blur', matching, categories)).toBe(blur);
});
