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

// The node sweep's runtime results (2026-10-23): base, overlay, position, x, y, crop -> output.
test.each([
    ['3x5x3', '9x17x3', 'Center', 0, 0, false, '9x17x4'],
    ['9x17x1', '3x5x3', 'Center', 0, 0, false, '9x17x3'],
    ['3x5x3', '3x5x3', 'PixelOffset', -1, 0, false, '4x5x4'],
    ['3x5x1', '3x5x3', 'PixelOffset', 100, 100, true, '3x5x1'],
    ['3x5x1', '3x5x3', 'PixelOffset', 2, 1, true, '3x5x3'],
    ['3x5x3', '3x5x4', 'PercentOffset', 0, 0, false, '3x5x4'],
])(
    'Blend Images of %s and %s at %s %s,%s (crop %s) declares %s',
    (base, overlay, position, x, y, crop, expected) => {
        const image = (size: string) => {
            const [width, height, channels] = size.split('x');
            return `Image { width: ${width}, height: ${height}, channels: ${channels} }`;
        };
        const result = outputType(
            'image_utility/compositing/blend_images.py',
            `enum BlendOverlayPosition { TopLeft, Top, TopRight, Left, Center, Right, BottomLeft,
                 Bottom, BottomRight, PercentOffset, PixelOffset }
             let Input0 = ${image(base)}; let Input1 = ${image(overlay)};
             let Input3 = BlendOverlayPosition::${position};
             let Input4 = ${x}; let Input5 = ${y}; let Input6 = ${x}; let Input7 = ${y};
             let Input8 = ${String(crop)};`
        );
        expect(isSameType(result, expression(image(expected)))).toBe(true);
    }
);

// Upstream failed to broadcast these (2026-10-23 node sweep); they keep their own size.
test.each([
    ['Horizontal', 3, 5, 22, 9],
    ['Vertical', 5, 3, 17, 19],
])(
    'Stack %s keeps a narrower-channel image at its own size',
    (orientation, w, h, width, height) => {
        const result = outputType(
            'image_utility/compositing/stack_images.py',
            `enum Orientation { Horizontal, Vertical }
         let Input0 = Image { width: ${w}, height: ${h}, channels: 1 };
         let Input1 = Image { width: 17, height: 9, channels: 4 };
         let Input2 = null; let Input3 = null; let Input4 = Orientation::${orientation};
         ${Array.from({ length: 10 }, (_, i) => `let Input${i + 5} = null;`).join('\n')}`
        );
        expect(
            isSameType(
                result,
                expression(`Image { width: ${width}, height: ${height}, channels: 4 }`)
            )
        ).toBe(true);
    }
);

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

// A connected number reaches the node rounded to its input's decimal places, as the input's
// conversion in numeric_inputs.py tells the types (backend/tests/test_number_input_rounding.py):
// Rotate turned by 180.0 while declaring the size of a 180.013-degree turn (2026-10-23 sweep).
const decimalPlaces = (scale: number) => {
    const source = readFileSync(
        path.resolve('backend/src/nodes/properties/inputs/numeric_inputs.py'),
        'utf-8'
    );
    const match = /InputConversion\("number", f"([^"]*)"\)/.exec(source);
    if (!match) throw new Error('Missing the decimal places conversion');
    return match[1].replace(/\{scale\}/g, String(scale));
};

test.each([
    [180.013, 3, 5],
    [89.96, 5, 3],
])('Rotate connected to %s declares the size of the rounded turn', (angle, width, height) => {
    const result = outputType(
        'image_utility/modification/rotate.py',
        `enum RotateSizeChange { Expand, Crop }
         enum FillColor { Auto, Black, Transparent }
         def FillColor::getOutputChannels(fill: FillColor, channels: uint) {
             match fill { FillColor::Transparent => 4, _ => channels }
         }
         let Input0 = Image { width: 3, height: 5, channels: 3 };
         let Input = ${angle};
         let Input1 = ${decimalPlaces(10)};
         let Input3 = RotateSizeChange::Expand; let Input4 = FillColor::Auto;`
    );
    expect(
        isSameType(result, expression(`Image { width: ${width}, height: ${height}, channels: 3 }`))
    ).toBe(true);
});

test.each([
    [180.013, 1, 180],
    [0.35, 1, 0.4],
    [-0.35, 1, -0.3],
    [2.675, 2, 2.68],
    [0.125, 2, 0.13],
    [12.3456, 3, 12.346],
])(
    'The decimal places conversion takes %s at %s places to %s, as enforce does',
    (value, places, rounded) => {
        const converted = expression(`{ let Input = ${value}; ${decimalPlaces(10 ** places)} }`);
        expect(converted).toEqual(literal(rounded));
    }
);

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
