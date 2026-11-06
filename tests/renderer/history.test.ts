import { Node } from 'reactflow';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { NodeData, SchemaId } from '../../src/common/common-types';
import { HistoryProvider } from '../../src/renderer/components/HistoryProvider';
import { mount } from './hookRunner';

const stubs = vi.hoisted(() => ({
    nodes: [] as unknown[],
    edges: [] as unknown[],
    nodeChanges: 0,
    chainLoads: 0,
    hotkeys: new Map<string, () => void>(),
}));

vi.mock('react', async (importOriginal) => ({
    ...(await importOriginal<typeof import('react')>()),
    ...(await import('./hookRunner')).fakeReact,
}));
vi.mock('react-dom', () => ({ flushSync: (fn: () => void) => fn() }));
vi.mock('reactflow', () => ({
    useReactFlow: () => ({ getNodes: () => stubs.nodes, getEdges: () => stubs.edges }),
}));
vi.mock('use-context-selector', () => ({
    useContext: () => ({
        changeNodes: (nodes: unknown[]) => {
            stubs.nodes = nodes;
            stubs.nodeChanges += 1;
        },
        changeEdges: (edges: unknown[]) => {
            stubs.edges = edges;
        },
        flushInputChanges: () => {},
    }),
    useContextSelector: (_context: unknown, select: (value: unknown) => unknown) =>
        select({ nodeChanges: stubs.nodeChanges, edgeChanges: 0, chainLoads: stubs.chainLoads }),
}));
vi.mock('../../src/renderer/contexts/GlobalNodeState', () => ({
    GlobalContext: {},
    GlobalVolatileContext: {},
}));
vi.mock('../../src/renderer/hooks/useHotkeys', () => ({
    useHotkeys: (keys: string, fn: () => void) => stubs.hotkeys.set(keys.split(',')[0], fn),
}));
vi.mock('../../src/renderer/hooks/useIpcRendererListener', () => ({
    useIpcRendererListener: () => {},
}));

const node = (id: string): Node<NodeData> => ({
    id,
    type: 'regularNode',
    position: { x: 0, y: 0 },
    data: { id, schemaId: 'test:history:node' as SchemaId, inputData: {} },
});
const ids = () => (stubs.nodes as Node<NodeData>[]).map((n) => n.id);
const undo = () => stubs.hotkeys.get('ctrl+z')!();
const redo = () => stubs.hotkeys.get('ctrl+y')!();
const edit = async (view: { render: () => void }, nodes: Node<NodeData>[]) => {
    stubs.nodes = nodes;
    stubs.nodeChanges += 1;
    view.render();
    await vi.advanceTimersByTimeAsync(300);
};

beforeEach(() => {
    vi.useFakeTimers();
    stubs.nodes = [];
    stubs.edges = [];
    stubs.nodeChanges = 0;
    stubs.chainLoads = 0;
});

afterEach(() => {
    vi.useRealTimers();
});

test('undo does not go back from an opened chain to the chain before it', async () => {
    const view = mount(() => HistoryProvider({ children: null }));
    await edit(view, [node('a')]);
    await edit(view, [node('a'), node('a2')]);

    // Opening a chain. Its file's path is now the save path: an undo here would save the chain
    // before it over that file.
    stubs.chainLoads += 1;
    await edit(view, [node('b')]);
    undo();
    view.render();
    expect(ids()).toEqual(['b']);

    // the edits of the opened chain undo as ever, down to the chain as opened
    await vi.advanceTimersByTimeAsync(300);
    await edit(view, [node('b'), node('b2')]);
    undo();
    view.render();
    expect(ids()).toEqual(['b']);
    await vi.advanceTimersByTimeAsync(300);
    undo();
    view.render();
    expect(ids()).toEqual(['b']);
    redo();
    view.render();
    expect(ids()).toEqual(['b', 'b2']);
});

test('undo does not go back from a new chain to the chain before it', async () => {
    const view = mount(() => HistoryProvider({ children: null }));
    await edit(view, [node('f')]);

    // New, then Save As: an undo here would save the chain before it over the new file
    stubs.chainLoads += 1;
    await edit(view, []);
    undo();
    view.render();

    expect(ids()).toEqual([]);
});

test('an undo right after opening, before the opened chain was committed, stays on it', async () => {
    const view = mount(() => HistoryProvider({ children: null }));
    await edit(view, [node('a')]);

    stubs.chainLoads += 1;
    stubs.nodes = [node('b')];
    stubs.nodeChanges += 1;
    view.render();
    undo();
    view.render();

    expect(ids()).toEqual(['b']);
});
