/**
 * The number that copied text is for a number input, in the input's precision as when typed: none
 * if the text is no finite number or outside of the input's limits.
 */
export const parsePastedNumber = (
    text: string,
    min: number | null | undefined,
    max: number | null | undefined,
    precision: number
): number | undefined => {
    const n = text.trim() ? Number(text) : NaN;
    if (!Number.isFinite(n) || (min != null && n < min) || (max != null && n > max)) {
        return undefined;
    }
    return Number((precision > 0 ? n : Math.round(n)).toFixed(precision));
};
