/**
 * No DOM packages are installed, so a component with hooks runs here as a plain function: `fakeReact`
 * replaces React's hooks (`vi.mock('react', ...)`) and `mount` calls the component again after each
 * batch of state changes, then runs the effects whose dependencies changed, as React does.
 */
interface Slot {
    value?: unknown;
    set?: (update: unknown) => void;
    deps?: readonly unknown[];
    cleanup?: void | (() => void);
}

let slots: Slot[] = [];
let cursor = 0;
let pendingEffects: (() => void)[] = [];
let dirty = false;
let scheduled = false;
let renderMounted: () => void = () => {};

const nextSlot = (): Slot => {
    const index = cursor;
    cursor += 1;
    slots[index] ??= {};
    return slots[index];
};
const sameDeps = (a?: readonly unknown[], b?: readonly unknown[]): boolean =>
    !!a && !!b && a.length === b.length && a.every((value, i) => Object.is(value, b[i]));

export const flush = (): void => {
    scheduled = false;
    for (let i = 0; dirty; i += 1) {
        if (i === 50) throw new Error('The component keeps changing its state');
        dirty = false;
        renderMounted();
    }
};

export const fakeReact = {
    memo: <T>(component: T): T => component,
    useState: <T>(initial: T | (() => T)) => {
        const slot = nextSlot();
        if (!slot.set) {
            slot.value = typeof initial === 'function' ? (initial as () => T)() : initial;
            slot.set = (update) => {
                const next =
                    typeof update === 'function'
                        ? (update as (old: unknown) => unknown)(slot.value)
                        : update;
                if (Object.is(next, slot.value)) return;
                slot.value = next;
                dirty = true;
                if (!scheduled) {
                    scheduled = true;
                    queueMicrotask(flush);
                }
            };
        }
        return [slot.value as T, slot.set] as const;
    },
    useRef: <T>(initial: T): { current: T } => {
        const slot = nextSlot();
        slot.value ??= { current: initial };
        return slot.value as { current: T };
    },
    useMemo: <T>(factory: () => T, deps: readonly unknown[]): T => {
        const slot = nextSlot();
        if (!sameDeps(slot.deps, deps)) {
            slot.deps = deps;
            slot.value = factory();
        }
        return slot.value as T;
    },
    useCallback: <T>(callback: T, deps: readonly unknown[]): T => {
        const slot = nextSlot();
        if (!sameDeps(slot.deps, deps)) {
            slot.deps = deps;
            slot.value = callback;
        }
        return slot.value as T;
    },
    useEffect: (effect: () => void | (() => void), deps?: readonly unknown[]): void => {
        const slot = nextSlot();
        if (!sameDeps(slot.deps, deps)) {
            slot.deps = deps;
            pendingEffects.push(() => {
                slot.cleanup?.();
                slot.cleanup = effect();
            });
        }
    },
};

/** Runs `component` as a mounted component. Returns its latest result, and how to render it again. */
export const mount = <T>(
    component: () => T
): { readonly result: T; readonly render: () => void } => {
    slots = [];
    pendingEffects = [];
    dirty = false;
    let result: T;
    renderMounted = () => {
        cursor = 0;
        result = component();
        const effects = pendingEffects;
        pendingEffects = [];
        effects.forEach((effect) => effect());
    };
    renderMounted();
    flush();
    return {
        get result() {
            return result;
        },
        render: () => {
            renderMounted();
            flush();
        },
    };
};
