import { ComponentProps, ReactElement, SetStateAction } from 'react';
import { Edge, Node } from 'reactflow';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { EdgeData, InputId, NodeData, SchemaId } from '../../src/common/common-types';
import { chainSnapshot } from '../../src/common/nodes/chainSnapshot';
import { AdvancedNumberInput } from '../../src/renderer/components/inputs/elements/AdvanceNumberInput';
import { GlobalContext, GlobalProvider } from '../../src/renderer/contexts/GlobalNodeState';
import type { SaveData } from '../../src/main/SaveFile';
import type { NumberInputProps } from '@chakra-ui/react';

// No DOM packages are installed. Keep refs/state between explicit component calls, and run
// only the numeric field's effects. Electron, React Flow and Chakra's DOM widgets are substitutes.
const harness = vi.hoisted(() => {
    const providerSlots: unknown[] = [];
    const inputSlots: unknown[] = [];
    const stateWrites = vi.fn<[unknown], void>();
    const state = {
        slots: providerSlots,
        cursor: 0,
        runEffects: false,
        cleanups: [] as (() => void)[],
        listeners: new Map<string, (...args: unknown[]) => unknown>(),
        nodes: [] as Node<NodeData>[],
        edges: [] as Edge<EdgeData>[],
        invoke: vi.fn<unknown[], Promise<unknown>>(),
        send: vi.fn(),
    };
    const useRef = <T>(initial: T): { current: T } => {
        const index = state.cursor;
        state.cursor += 1;
        if (index === state.slots.length) state.slots.push({ current: initial });
        return state.slots[index] as { current: T };
    };
    const useState = <T>(initial: T | (() => T)) => {
        const ref = useRef<{ value: T } | undefined>(undefined);
        ref.current ??= { value: typeof initial === 'function' ? (initial as () => T)() : initial };
        const cell = ref.current;
        return [
            cell.value,
            (update: SetStateAction<T>) => {
                cell.value =
                    typeof update === 'function' ? (update as (old: T) => T)(cell.value) : update;
                stateWrites(cell.value);
            },
        ] as const;
    };
    const flow = {
        getNodes: () => state.nodes,
        getEdges: () => state.edges,
        getNode: (id: string) => state.nodes.find((n) => n.id === id),
        getViewport: () => ({ x: 0, y: 0, zoom: 1 }),
        setNodes: (update: SetStateAction<Node<NodeData>[]>) => {
            state.nodes = typeof update === 'function' ? update(state.nodes) : update;
        },
        setEdges: (update: SetStateAction<Edge<EdgeData>[]>) => {
            state.edges = typeof update === 'function' ? update(state.edges) : update;
        },
        setViewport: vi.fn(),
    };
    return { state, providerSlots, inputSlots, stateWrites, useRef, useState, flow };
});

vi.mock('react', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react')>();
    return {
        ...actual,
        memo: <T>(component: T) => component,
        useRef: harness.useRef,
        useState: harness.useState,
        useCallback: <T>(callback: T) => callback,
        useMemo: <T>(factory: () => T) => factory(),
        useEffect: (effect: () => void | (() => void)) => {
            if (harness.state.runEffects) {
                const cleanup = effect();
                if (cleanup) harness.state.cleanups.push(cleanup);
            }
        },
    };
});
vi.mock('react-dom', () => ({ flushSync: (callback: () => void) => callback() }));
vi.mock('reactflow', () => ({
    useReactFlow: () => harness.flow,
    useViewport: () => ({ x: 0, y: 0, zoom: 1 }),
}));
vi.mock('use-context-selector', () => ({
    createContext: <T>(value: T) => ({ value, Provider: 'context-provider' }),
    useContext: <T>(context: { value: T }) => context.value,
    useContextSelector: <T, R>(context: { value: T }, selector: (value: T) => R) =>
        selector(context.value),
}));
vi.mock('@chakra-ui/react', () => ({
    InputGroup: 'input-group',
    InputLeftAddon: 'input-left-addon',
    NumberInput: 'number-input',
    NumberInputField: 'number-input-field',
    NumberInputStepper: 'number-input-stepper',
    NumberIncrementStepper: 'number-increment-stepper',
    NumberDecrementStepper: 'number-decrement-stepper',
}));
vi.mock('../../src/renderer/contexts/AlertBoxContext', () => ({
    AlertBoxContext: {
        value: {
            sendAlert: vi.fn(),
            sendToast: vi.fn(),
            showAlert: vi.fn(() => Promise.resolve(1)),
        },
    },
    AlertType: { WARN: 'warn', ERROR: 'error' },
}));
vi.mock('../../src/renderer/contexts/BackendContext', () => ({
    BackendContext: {
        value: {
            schemata: {
                has: () => true,
                get: () => ({ deprecated: false }),
                getDefaultInput: () => ({}),
            },
            functionDefinitions: new Map(),
            passthrough: new Map(),
            backend: {},
            nodesFromCache: false,
        },
    },
}));
vi.mock('../../src/renderer/hooks/useSettings', () => ({ useSettings: () => ({}) }));
vi.mock('../../src/renderer/hooks/useOpenRecent', () => ({
    useOpenRecent: () => [[], vi.fn(), vi.fn()],
}));
vi.mock('../../src/renderer/hooks/useInputHashes', () => ({
    useInputHashes: () => [new Map(), { current: new Map() }],
}));
vi.mock('../../src/renderer/hooks/useAsyncEffect', () => ({ useAsyncEffect: () => {} }));
vi.mock('../../src/renderer/hooks/useHotkeys', () => ({ useHotkeys: () => {} }));
vi.mock('../../src/renderer/hooks/useIpcRendererListener', () => ({
    useIpcRendererListener: (channel: string, listener: (...args: unknown[]) => unknown) => {
        harness.state.listeners.set(channel, listener);
    },
}));
vi.mock('../../src/renderer/safeIpc', () => ({
    ipcRenderer: { invoke: harness.state.invoke, send: harness.state.send },
}));

const emit = async (channel: string, ...args: unknown[]): Promise<void> => {
    const listener = harness.state.listeners.get(channel);
    expect(listener, channel).toBeDefined();
    await listener!(undefined, ...args);
};

const renderProvider = () => {
    harness.state.slots = harness.providerSlots;
    harness.state.cursor = 0;
    harness.state.runEffects = false;
    const tree = GlobalProvider({ reactFlowWrapper: { current: null } }) as ReactElement<{
        children: ReactElement<{ value: ComponentProps<typeof GlobalContext.Provider>['value'] }>[];
    }>;
    const global = tree.props.children[0].props.value;
    Object.assign(GlobalContext, { value: global });
    return global;
};

beforeEach(() => {
    harness.providerSlots.length = 0;
    harness.inputSlots.length = 0;
    harness.state.listeners.clear();
    harness.stateWrites.mockClear();
    harness.state.invoke.mockReset().mockResolvedValue(undefined);
    harness.state.send.mockClear();
    harness.state.nodes = [
        {
            id: 'number',
            type: 'regularNode',
            position: { x: 0, y: 0 },
            data: {
                id: 'number',
                schemaId: 'chainner:utility:number' as SchemaId,
                inputData: { [0 as InputId]: 1 },
            },
        },
    ];
    harness.state.edges = [];
    vi.stubGlobal('sessionStorage', {
        getItem: (key: string) =>
            key === 'save-path' ? JSON.stringify('C:/chains/current.chn') : null,
        setItem: vi.fn(),
    });
});
afterEach(() => {
    harness.state.cleanups.splice(0).forEach((cleanup) => cleanup());
    vi.unstubAllGlobals();
});

test.each(['typed', 'NumberInput change'])(
    '%s draft 2 is committed before the Ctrl+S file-save handler runs',
    async (kind) => {
        const global = renderProvider();
        const setInput = vi.fn((value: number) =>
            global.setNodeInputValue('number', 0 as InputId, value)
        );
        let inputString = '1';
        const renderInput = () => {
            harness.state.slots = harness.inputSlots;
            harness.state.cursor = 0;
            harness.state.runEffects = true;
            const tree = AdvancedNumberInput({
                min: 0,
                max: 10,
                precision: 0,
                controlsStep: 1,
                defaultValue: 1,
                hideTrailingZeros: true,
                inputString,
                setInput,
                setInputString: (value) => {
                    inputString = value;
                },
            }) as ReactElement<{ children: ReactElement<NumberInputProps>[] }>;
            return tree.props.children[1];
        };
        const numberInput = renderInput();
        if (kind === 'typed') {
            const field = (
                numberInput.props.children as ReactElement<{ onChange?: () => void }>[]
            )[0];
            field.props.onChange?.();
        }
        numberInput.props.onChange!('2', 2);
        renderInput();
        expect(setInput).not.toHaveBeenCalled();

        // main-window's CmdOrCtrl+S menu accelerator sends this event to GlobalProvider.
        await emit('file-save');
        expect(setInput).toHaveBeenCalledOnce();
        expect(setInput).toHaveBeenCalledWith(2);
        const [channel, savedData, savePath] = harness.state.invoke.mock.calls[0];
        expect(channel).toBe('file-save-json');
        expect(savePath).toBe('C:/chains/current.chn');
        expect((savedData as SaveData).nodes[0].data.inputData[0 as InputId]).toBe(2);
    }
);

test.each(['file-new', 'file-open'])(
    'a delayed Save As cannot update the document after %s',
    async (channel) => {
        renderProvider();
        let finish: (result: unknown) => void = () => {};
        const pending = new Promise<unknown>((resolve) => {
            finish = resolve;
        });
        harness.state.invoke.mockReturnValueOnce(pending);
        const save = emit('file-save-as');
        await emit(channel, {
            kind: 'Success',
            path: 'C:/chains/other.chn',
            saveData: {
                nodes: [],
                edges: [],
                viewport: { x: 0, y: 0, zoom: 1 },
                tamperedWith: false,
            },
        });
        await vi.waitFor(() => expect(harness.state.nodes).toEqual([]));
        const changes = [...harness.stateWrites.mock.calls];
        finish({ kind: 'Success', path: 'C:/chains/old-save-as.chn' });
        await save;
        expect(harness.stateWrites.mock.calls).toEqual(changes);
    }
);

test('an edit during Save keeps the snapshot of the data that was actually sent', async () => {
    const global = renderProvider();
    const snapshot = chainSnapshot(harness.state.nodes, harness.state.edges);
    let finish = () => {};
    const pending = new Promise<void>((resolve) => {
        finish = resolve;
    });
    harness.state.invoke.mockReturnValueOnce(pending);
    const save = emit('file-save');
    global.setNodeInputValue('number', 0 as InputId, 2);
    finish();
    await save;
    expect(harness.stateWrites).toHaveBeenLastCalledWith(snapshot);
    expect(chainSnapshot(harness.state.nodes, harness.state.edges)).not.toBe(snapshot);
});
