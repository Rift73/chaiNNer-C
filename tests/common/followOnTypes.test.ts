import {
    SourceDocument,
    evaluate,
    intersect,
    isDisjointWith,
    isSameType,
    isSubsetOf,
    literal,
    parseExpression,
} from '@chainner/navi';
import { readFileSync } from 'fs';
import path from 'path';
import { expect, test } from 'vitest';
import { getChainnerScope } from '../../src/common/types/chainner-scope';

const expression = (code: string) =>
    evaluate(parseExpression(new SourceDocument(code, 'follow-on types')), getChainnerScope());

const outputType = (file: string, bindings: string) => {
    const source = readFileSync(
        path.resolve('backend/src/packages/chaiNNer_standard', file),
        'utf-8'
    );
    const match = /(?:output_type|image_type)="""([\s\S]*?)"""|output_type="([^"]+)"/.exec(source);
    if (!match) throw new Error(`Missing output type in ${file}`);
    return expression(`{ ${bindings}\n${match[1] || match[2]} }`);
};

test.each([
    [2, 8, 3],
    [1, 2, undefined],
])('Math log base %s of %s has the runtime type', (base, value, expected) => {
    const result = outputType(
        'utility/math/math.py',
        `enum MathOperation { Add, Subtract, Multiply, Divide, Power, Log, Maximum, Minimum, Modulo, Percent }
         let Input0 = ${base}; let Input1 = MathOperation::Log; let Input2 = ${value};`
    );
    const number = intersect(result, expression('number'));
    if (expected === undefined) {
        expect(number.type).toBe('never');
        expect(result.type).toBe('instance');
        expect(isSubsetOf(result, expression('Error'))).toBe(true);
    } else {
        expect(isSameType(number, literal(expected))).toBe(true);
    }
});

test('center padding follows CPython for an odd margin and odd width', () => {
    expect(expression('padCenter("AB", 3, "_")')).toEqual(literal('_AB'));
});

test.each([
    ['Horizontal', 1, 4, 8, 4],
    ['Vertical', 4, 1, 4, 8],
])('Stack %s includes the expanded one-pixel strip', (orientation, w, h, width, height) => {
    const result = outputType(
        'image_utility/compositing/stack_images.py',
        `enum Orientation { Horizontal, Vertical }
         let Input0 = Image { width: ${w}, height: ${h}, channels: 1 };
         let Input1 = Image { width: 4, height: 4, channels: 4 };
         let Input2 = null; let Input3 = null; let Input4 = Orientation::${orientation};
         ${Array.from({ length: 10 }, (_, i) => `let Input${i + 5} = null;`).join('\n')}`
    );
    expect(
        isSameType(result, expression(`Image { width: ${width}, height: ${height}, channels: 4 }`))
    ).toBe(true);
});

test.each([
    [2, 3],
    [3, 2],
    [3, 3],
])('Crop Border permits the %s by %s edge/interior result', (width, height) => {
    const result = outputType(
        'image_dimension/crop/crop_border.py',
        'let Input0 = Image { width: 5, height: 5, channels: 3 }; let Input3 = 1;'
    );
    expect(
        isSubsetOf(expression(`Image { width: ${width}, height: ${height}, channels: 3 }`), result)
    ).toBe(true);
    expect(isDisjointWith(expression('Image { width: 1, height: 1, channels: 3 }'), result)).toBe(
        true
    );
});

test.each([
    ['1_000', 1000],
    ['1__0', undefined],
])('Parse Number models the underscore grammar of %s', (text, expected) => {
    const result = outputType(
        'utility/value/parse_number.py',
        `let Input0 = "${text}"; let Input1 = 10;`
    );
    if (expected === undefined) {
        expect(result.type).toBe('never');
    } else {
        expect(result).toEqual(literal(expected));
    }
});
