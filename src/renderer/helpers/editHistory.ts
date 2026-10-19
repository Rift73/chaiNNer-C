/**
 * A variable-length linear edit history.
 *
 * The linear history is stored as a non-empty array with an index pointing to the current value.
 * The history array represents a linear timeline such if value A was before value B in time, then
 * value A will have a smaller index than value B.
 *
 * Every value carries a `key`; values with the same key are the same step, so committing a value
 * with the key of the current value changes nothing (and keeps the redo steps).
 */
export class EditHistory<T extends { readonly key: string }> {
    private readonly history: readonly T[];

    private readonly currentIndex: number;

    readonly maxLength: number;

    get current(): T {
        return this.history[this.currentIndex];
    }

    private constructor(history: readonly T[], currentIndex: number, maxLength: number) {
        if (history.length === 0) {
            throw new Error('Invalid history: History is empty');
        }
        if (currentIndex < 0 || currentIndex >= history.length) {
            throw new Error('Invalid history: Current index out of bounds');
        }

        this.history = history;
        this.currentIndex = currentIndex;
        this.maxLength = maxLength;
    }

    static create<T extends { readonly key: string }>(
        initialValue: T,
        maxLength: number
    ): EditHistory<T> {
        return new EditHistory([initialValue], 0, maxLength);
    }

    undo(): EditHistory<T> {
        if (this.currentIndex === 0) return this;
        return new EditHistory(this.history, this.currentIndex - 1, this.maxLength);
    }

    redo(): EditHistory<T> {
        if (this.currentIndex === this.history.length - 1) return this;
        return new EditHistory(this.history, this.currentIndex + 1, this.maxLength);
    }

    commit(value: T): EditHistory<T> {
        if (value.key === this.current.key) return this;
        const start = this.currentIndex + 1 < this.maxLength ? 0 : 1;
        const newHistory = this.history.slice(start, this.currentIndex + 1);
        newHistory.push(value);
        return new EditHistory(newHistory, newHistory.length - 1, this.maxLength);
    }
}
