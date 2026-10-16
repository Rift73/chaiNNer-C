import { expect, test } from 'vitest';
import { IdSet } from '../../src/common/IdSet';

test('round-trips ids 0, 65535, 65536, 2147483647', () => {
    const ids = [0, 65535, 65536, 2147483647];
    const set = IdSet.from(ids);

    expect([...IdSet.toSet(set)]).toStrictEqual(ids);
    for (const id of ids) {
        expect(IdSet.has(set, id)).toBe(true);
    }
    expect(IdSet.has(set, 1)).toBe(false);
    expect(IdSet.has(set, 65537)).toBe(false);
    expect(IdSet.isEmpty(set)).toBe(false);
    expect(IdSet.isEmpty(IdSet.from([]))).toBe(true);
});

test('has does not match across pair boundaries', () => {
    // 1 and 2*65536 are stored as the pairs (0, 1) and (2, 0); read across the boundary,
    // the units in the middle form the pair (1, 2), which is the id 65536 + 2.
    const set = IdSet.from([1, 65536 * 2]);
    expect(IdSet.has(set, 65536 * 1 + 2)).toBe(false);
    expect(IdSet.toSet(set)).toStrictEqual(new Set([1, 65536 * 2]));

    // every pair that straddles a boundary of these sets must not be found
    const sets = [
        [1, 65536 * 2],
        [65536 + 1, 65536 * 3 + 7, 2147483647],
        [0, 1, 2, 65535, 65536, 65537],
    ];
    for (const ids of sets) {
        const idSet = IdSet.from(ids);
        for (let i = 1; i + 1 < idSet.length; i += 2) {
            const straddling = idSet.charCodeAt(i) * 0x10000 + idSet.charCodeAt(i + 1);
            if (!ids.includes(straddling)) {
                expect(IdSet.has(idSet, straddling)).toBe(false);
            }
        }
    }
});

test('equal sets give equal strings', () => {
    expect(IdSet.from([5, 70000, 5, 1])).toBe(IdSet.from([1, 70000, 5]));
    expect(IdSet.from(new Set([2147483647, 0]))).toBe(IdSet.from([0, 2147483647, 0]));
    expect(IdSet.from([1, 2])).not.toBe(IdSet.from([1, 3]));
});

test('survives the JSON round trip of nodeState', () => {
    // low halves 0xD800 and 0xDFFF are lone surrogates; well-formed JSON.stringify escapes them
    const ids = [0xd800, 0x1dfff, 1000, 2147483647];
    const sets = [IdSet.from(ids), IdSet.from([1])] as const;
    const parsed = JSON.parse(JSON.stringify(sets)) as typeof sets;
    expect(parsed).toStrictEqual(sets);
    expect(IdSet.toSet(parsed[0])).toStrictEqual(new Set([1000, 0xd800, 0x1dfff, 2147483647]));
});

test('rejects ids outside 0..2147483647', () => {
    expect(() => IdSet.from([-1])).toThrow();
    expect(() => IdSet.from([2147483648])).toThrow();
    expect(() => IdSet.from([1.5])).toThrow();
});
