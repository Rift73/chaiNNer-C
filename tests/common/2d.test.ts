import { expect, test } from 'vitest';
import { Vec2 } from '../../src/common/2d';

test('a vector is multiplied and divided by each component of another', () => {
    expect(new Vec2(2, 3).mul(new Vec2(4, 5))).toEqual(new Vec2(8, 15));
    expect(new Vec2(8, 15).div({ x: 4, y: 5 })).toEqual(new Vec2(2, 3));
    expect(new Vec2(2, 3).mul(2)).toEqual(new Vec2(4, 6));
});
