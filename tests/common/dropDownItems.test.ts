import { expect, test } from 'vitest';
import { DropdownGroup, InputOption } from '../../src/common/common-types';
import {
    DropDownItem,
    filterDropDownItems,
    getDropDownItems,
    getUnavailableMask,
} from '../../src/renderer/helpers/dropDownItems';

const options: readonly InputOption[] = [
    { option: 'BM3D', value: 'vsdenoise.bm3d' },
    { option: 'NL-Means', value: 'vsdenoise.nl_means' },
    { option: 'BM3D CUDA', value: 'bm3dcuda.BM3D' },
    { option: 'DFTTest', value: 'dfttest.DFTTest' },
    { option: 'Remove Grain', value: 'rgvs.RemoveGrain' },
    { option: 'Median', value: 'std.Median' },
];
const groups: readonly DropdownGroup[] = [
    { label: 'Common', startAt: 'vsdenoise.bm3d' },
    { label: 'All', startAt: 'bm3dcuda.BM3D' },
    { label: null, startAt: 'rgvs.RemoveGrain' },
];

const describe = (items: readonly DropDownItem[]): string[] =>
    items.map((item) =>
        item.kind === 'group' ? `[${item.group.label ?? '---'}]` : item.option.option
    );

test('items list options in order with their group headers', () => {
    const items = getDropDownItems(options, groups);
    expect(describe(items)).toStrictEqual([
        '[Common]',
        'BM3D',
        'NL-Means',
        '[All]',
        'BM3D CUDA',
        'DFTTest',
        '[---]',
        'Remove Grain',
        'Median',
    ]);
    // option items keep their index into `options`
    expect(items.flatMap((i) => (i.kind === 'option' ? [i.index] : []))).toStrictEqual([
        0, 1, 2, 3, 4, 5,
    ]);
});

test('dropdown filters options by typed text', () => {
    const items = getDropDownItems(options, groups);

    // empty or blank queries keep everything (the same list, no copy)
    expect(filterDropDownItems(items, '')).toBe(items);
    expect(filterDropDownItems(items, '   ')).toBe(items);

    // case-insensitive substring of the label; headers only above groups with a match
    expect(describe(filterDropDownItems(items, 'bm3d'))).toStrictEqual([
        '[Common]',
        'BM3D',
        '[All]',
        'BM3D CUDA',
    ]);
    expect(describe(filterDropDownItems(items, 'nl-ME'))).toStrictEqual(['[Common]', 'NL-Means']);

    // every whitespace-separated term must match
    expect(describe(filterDropDownItems(items, 'cuda bm'))).toStrictEqual(['[All]', 'BM3D CUDA']);

    // a separator is kept between matches but never leads the list
    expect(describe(filterDropDownItems(items, 'me'))).toStrictEqual([
        '[Common]',
        'NL-Means',
        '[---]',
        'Median',
    ]);
    expect(describe(filterDropDownItems(items, 'median'))).toStrictEqual(['Median']);
    expect(describe(filterDropDownItems(items, 'e'))).toStrictEqual([
        '[Common]',
        'NL-Means',
        '[All]',
        'DFTTest',
        '[---]',
        'Remove Grain',
        'Median',
    ]);

    // option values are not searched, and no match gives an empty list
    expect(filterDropDownItems(items, 'vsdenoise')).toStrictEqual([]);
});

test('unavailable mask is stable text and empty without conditions', () => {
    expect(getUnavailableMask(options, () => false)).toBe('');
    expect(getUnavailableMask(options, undefined)).toBe('');

    const conditional: readonly InputOption[] = [
        { option: 'A', value: 0 },
        {
            option: 'B',
            value: 1,
            condition: { kind: 'not', condition: { kind: 'and', items: [] } },
        },
        { option: 'C', value: 2, condition: { kind: 'and', items: [] } },
    ];
    expect(getUnavailableMask(conditional, (c) => c.kind === 'and')).toBe('010');
});
