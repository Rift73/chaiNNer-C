import { Condition, DropdownGroup, InputOption } from '../../common/common-types';
import { EMPTY_ARRAY } from '../../common/util';

export interface DropDownGroupItem {
    readonly kind: 'group';
    /** A labeled group shows a header, an unlabeled group a separator line. */
    readonly group: DropdownGroup;
    /** The index of the option that starts the group. */
    readonly index: number;
}
export interface DropDownOptionItem {
    readonly kind: 'option';
    readonly option: InputOption;
    /** The index of the option in the input's options. */
    readonly index: number;
    /** The lower-case label used for searching. */
    readonly search: string;
}
export type DropDownItem = DropDownGroupItem | DropDownOptionItem;

/**
 * Returns the options of a dropdown in order, each group item placed before the option it starts at.
 */
export const getDropDownItems = (
    options: readonly InputOption[],
    groups: readonly DropdownGroup[] = EMPTY_ARRAY
): readonly DropDownItem[] => {
    const groupStarts = new Map<InputOption['value'], DropdownGroup>();
    for (const group of groups) {
        if (!groupStarts.has(group.startAt)) {
            groupStarts.set(group.startAt, group);
        }
    }

    const items: DropDownItem[] = [];
    options.forEach((option, index) => {
        const group = groupStarts.get(option.value);
        if (group) {
            items.push({ kind: 'group', group, index });
        }
        items.push({ kind: 'option', option, index, search: option.option.toLowerCase() });
    });
    return items;
};

/**
 * Keeps the options whose label contains every whitespace-separated term of the query (ignoring
 * case). A group item is kept only if one of its options is kept, and a separator never starts the
 * list. An empty query returns the given list.
 */
export const filterDropDownItems = (
    items: readonly DropDownItem[],
    query: string
): readonly DropDownItem[] => {
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (terms.length === 0) return items;

    const result: DropDownItem[] = [];
    let pendingGroup: DropDownGroupItem | undefined;
    for (const item of items) {
        if (item.kind === 'group') {
            pendingGroup = item;
        } else if (terms.every((term) => item.search.includes(term))) {
            if (pendingGroup && (pendingGroup.group.label || result.length > 0)) {
                result.push(pendingGroup);
            }
            pendingGroup = undefined;
            result.push(item);
        }
    }
    return result;
};

/**
 * Returns one character per option, `'1'` if the option's condition is not met and `'0'` otherwise,
 * or an empty string if no option has a condition. A string compares by value, so it stays equal
 * across edits that don't change which options are available.
 */
export const getUnavailableMask = (
    options: readonly InputOption[],
    testCondition: ((condition: Condition) => boolean) | undefined
): string => {
    if (!testCondition || options.every((o) => o.condition == null)) return '';
    return options
        .map((o) => (o.condition != null && !testCondition(o.condition) ? '1' : '0'))
        .join('');
};
