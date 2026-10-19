import { StringType, Type, literal } from '@chainner/navi';
import { expect, test } from 'vitest';
import { setTypeInMap } from '../../src/renderer/hooks/useTypeMap';

const newMap = () => new Map<string, Map<number, Type>>();

test('a new type changes the map', () => {
    const map = newMap();

    expect(setTypeInMap(map, 'a', 0, literal(2))).toBe(true);
    expect(map.get('a')?.get(0)).toEqual(literal(2));

    expect(setTypeInMap(map, 'a', 1, StringType.instance)).toBe(true);
    expect(setTypeInMap(map, 'a', 0, literal(3))).toBe(true);
    expect(map.get('a')?.get(0)).toEqual(literal(3));
});

test('the same type again is no change and keeps the stored type', () => {
    const map = newMap();
    const first = literal(2);
    setTypeInMap(map, 'a', 0, first);

    expect(setTypeInMap(map, 'a', 0, literal(2))).toBe(false);
    expect(map.get('a')?.get(0)).toBe(first);
});

test('removing a type changes the map only if the output had one', () => {
    const map = newMap();

    expect(setTypeInMap(map, 'a', 0, undefined)).toBe(false);
    setTypeInMap(map, 'a', 0, literal(2));
    expect(setTypeInMap(map, 'a', 0, undefined)).toBe(true);
    expect(map.get('a')?.has(0)).toBe(false);
    expect(setTypeInMap(map, 'a', 0, undefined)).toBe(false);
});
