import { CategoryMap } from '../../common/CategoryMap';
import { NodeSchema } from '../../common/common-types';
import { RRegex } from '../../common/rust-regex';
import { EMPTY_ARRAY, lazy } from '../../common/util';

const isLetter = lazy(() => new RRegex('(?is)^[a-z]$'));
const createSearchPredicate = (query: string): ((name: string) => boolean) => {
    if (!query) return () => true;

    const pattern = new RRegex(
        `(?is)^${[...query]
            .map((char) => {
                const hex = `\\u{${char.codePointAt(0)!.toString(16)}}`;
                const before = isLetter().isMatch(char) ? `[^a-z]` : `.`;
                return `(?:.*${before})?${hex}`;
            })
            .join('')}`
    );
    return (name) => pattern.isMatch(name);
};

export const getMatchingNodes = (
    searchQuery: string,
    schemata: readonly NodeSchema[],
    categories: CategoryMap
) => {
    const matchesSearchQuery = createSearchPredicate(searchQuery);
    const matchingNodes = !searchQuery
        ? schemata
        : schemata.filter((n) => {
              const category = categories.get(n.category)?.name ?? n.category;
              const nodeGroup = categories.getGroup(n.nodeGroup)?.name ?? n.nodeGroup;
              return (
                  matchesSearchQuery(`${category} ${n.name}`) ||
                  matchesSearchQuery(`${nodeGroup} ${n.name}`) ||
                  (n.keywords?.some((keyword) => matchesSearchQuery(keyword)) ?? false)
              );
          });

    return matchingNodes;
};

const getMax = <T extends NonNullable<unknown>>(
    iter: Iterable<T>,
    selector: (t: T) => number
): T | undefined => {
    let max: T | undefined;
    let maxVal = -Infinity;
    for (const item of iter) {
        const val = selector(item);
        if (val > maxVal) {
            maxVal = val;
            max = item;
        }
    }
    return max;
};

export const getBestMatch = (
    searchQuery: string,
    matchingSchemata: readonly NodeSchema[],
    _categories: CategoryMap,
    scoreMultiplier: (schema: NodeSchema) => number = () => 1
): NodeSchema | undefined => {
    // eslint-disable-next-line no-param-reassign
    searchQuery = searchQuery.trim().toLowerCase();
    if (searchQuery.length <= 1) {
        // there's no point in matching against super short queries
        return undefined;
    }

    const g2Points = 1;
    const g3Points = 3;
    const g4Points = 10;

    const letter = isLetter();
    const isBoundary = (s: string, index: number) => {
        if (index === 0) return true;
        const before = s[index - 1];
        return !letter.isMatch(before);
    };
    interface Matches {
        matches: number;
        boundaryMatches: number;
    }
    const countNGramMatches = (name: string, n: number): Matches => {
        let matches = 0;
        let boundaryMatches = 0;
        for (let i = 0; i <= searchQuery.length - n; i += 1) {
            const index = name.indexOf(searchQuery.slice(i, i + n));
            if (index !== -1) {
                if (isBoundary(name, index)) {
                    boundaryMatches += 1;
                } else {
                    matches += 1;
                }
            }
        }
        return { matches, boundaryMatches };
    };

    const scoreMatches = (matches: Matches, basePoints: number) => {
        return matches.matches * basePoints + matches.boundaryMatches * basePoints * 3;
    };

    const scoreText = (text: string): number => {
        if (text.length === 0) return 0;
        const points =
            scoreMatches(countNGramMatches(text, 2), g2Points) +
            scoreMatches(countNGramMatches(text, 3), g3Points) +
            scoreMatches(countNGramMatches(text, 4), g4Points);
        return points / text.length;
    };

    // a keyword match counts a little less than an equally good name match
    const keywordWeight = 0.9;

    return getMax(matchingSchemata, (schema) => {
        let score = scoreText(schema.name.toLowerCase());
        for (const keyword of schema.keywords ?? EMPTY_ARRAY) {
            score = Math.max(score, scoreText(keyword.toLowerCase()) * keywordWeight);
        }

        return score * scoreMultiplier(schema);
    });
};
