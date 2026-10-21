import { Expression, Scope, Type, evaluate, isSameType } from '@chainner/navi';
import { useCallback, useState } from 'react';
import { log } from '../../common/log';

/**
 * Sets the type of an output in the map, or removes it if `type` is undefined. Returns whether
 * the map changed: a type that is the same as the one the output has is no change.
 */
export const setTypeInMap = <N, I>(
    map: Map<N, Map<I, Type>>,
    nodeId: N,
    outputId: I,
    type: Type | undefined
): boolean => {
    const inner = map.get(nodeId);
    if (!type) {
        return inner?.delete(outputId) ?? false;
    }
    const current = inner?.get(outputId);
    if (current && isSameType(current, type)) {
        return false;
    }
    if (inner) {
        inner.set(outputId, type);
    } else {
        map.set(nodeId, new Map([[outputId, type]]));
    }
    return true;
};

/**
 * A map of types that can be used as either a ref-like object or a state-like value.
 *
 * The state only changes when the map does, so setting the types an output already has does not
 * re-render or re-run the type pass.
 */
export const useTypeMap = <N, I>(scope: Scope) => {
    const [types, setTypes] = useState(() => ({
        map: new Map<N, Map<I, Type>>(),
    }));
    // the map itself stays the same object for the whole life of the component
    const { map } = types;

    const setType = useCallback(
        (nodeId: N, outputId: I, expression: Expression | undefined): void => {
            const getType = () => {
                if (expression === undefined) {
                    return undefined;
                }

                try {
                    return evaluate(expression, scope);
                } catch (error) {
                    log.error(error);
                    return undefined;
                }
            };

            if (setTypeInMap(map, nodeId, outputId, getType())) {
                setTypes({ map });
            }
        },
        [setTypes, map, scope]
    );

    const clear = useCallback(
        (nodes: Iterable<N>): void => {
            let changed = false;
            for (const nodeId of nodes) {
                changed = map.delete(nodeId) || changed;
            }
            if (changed) {
                setTypes({ map });
            }
        },
        [setTypes, map]
    );

    return [types, setType, clear] as const;
};
