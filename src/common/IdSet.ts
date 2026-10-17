/**
 * A value-type representation of a set of numeric ids. The main purpose of this type is to
 * provide a representation of sets that can be memoized by React.
 *
 * Ids must be integers in the range 0 to 2^31-1 (0x7FFFFFFF). Each id is stored as a fixed pair of
 * UTF-16 code units (high 16 bits, then low 16 bits), and the pairs are sorted and deduplicated,
 * so equal sets always give equal strings.
 */
export type IdSet<T extends number> = string & { __inputIdSet: never; __type: T };

// eslint-disable-next-line @typescript-eslint/no-namespace, @typescript-eslint/no-redeclare
export namespace IdSet {
    export const empty: IdSet<never> = '' as IdSet<never>;

    const isValidId = (id: number): boolean => Number.isInteger(id) && id >= 0 && id <= 0x7fffffff;

    const encode = (id: number): string =>
        String.fromCharCode(Math.floor(id / 0x10000), id % 0x10000);

    export const isEmpty = <T extends number>(idSet: IdSet<T>): boolean => {
        return idSet.length === 0;
    };

    export const has = <T extends number>(idSet: IdSet<T>, id: T): boolean => {
        if (!isValidId(id)) return false;
        const pair = encode(id);
        // only pairs that start at an even index are ids; others straddle two ids
        for (let i = idSet.indexOf(pair); i !== -1; i = idSet.indexOf(pair, i + 1)) {
            if (i % 2 === 0) return true;
        }
        return false;
    };
    export const toSet = <T extends number>(idSet: IdSet<T>): Set<T> => {
        const set = new Set<T>();
        for (let i = 0; i < idSet.length; i += 2) {
            set.add((idSet.charCodeAt(i) * 0x10000 + idSet.charCodeAt(i + 1)) as T);
        }
        return set;
    };

    const fromSorted = <T extends number>(sorted: readonly T[]): IdSet<T> => {
        let s = '';
        let last = -1;
        for (const id of sorted) {
            if (!isValidId(id)) throw new Error(`Invalid id: ${id}`);

            if (id !== last) {
                last = id;
                s += encode(id);
            }
        }
        return s as IdSet<T>;
    };
    export const from = <T extends number>(iter: Iterable<T>): IdSet<T> => {
        const sorted = Array.from(iter).sort((a, b) => a - b);
        return fromSorted(sorted);
    };
}
