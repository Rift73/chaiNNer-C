import { FunctionCallExpression, NamedExpression, evaluate, isDisjointWith } from '@chainner/navi';
import { expect, test } from 'vitest';
import { getChainnerScope } from '../../src/common/types/chainner-scope';
import { assertNever } from '../../src/common/util';

test(`Chainner scope is correct`, () => {
    const scope = getChainnerScope();

    for (const [name, def] of scope.entries()) {
        switch (def.type) {
            case 'parameter':
                break;
            case 'variable':
            case 'struct':
                evaluate(new NamedExpression(name), scope);
                break;
            case 'intrinsic-function':
            case 'function':
                evaluate(
                    new FunctionCallExpression(
                        name,
                        def.definition.parameters.map((p) => p.type)
                    ),
                    scope
                );
                break;
            default:
                return assertNever(def);
        }
    }
});

test(`VapourSynth clip and filter types are distinct`, () => {
    const scope = getChainnerScope();
    const clip = evaluate(new NamedExpression('VideoClip'), scope);
    const filter = evaluate(new NamedExpression('VSFilter'), scope);
    const video = evaluate(new NamedExpression('Video'), scope);

    expect(isDisjointWith(clip, filter)).toBe(true);
    expect(isDisjointWith(clip, video)).toBe(true);
    expect(isDisjointWith(filter, video)).toBe(true);
});
