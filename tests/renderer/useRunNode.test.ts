import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { InputId, NodeData, SchemaId } from '../../src/common/common-types';
import { useRunNode } from '../../src/renderer/hooks/useRunNode';
import { mount } from './hookRunner';

const stubs = vi.hoisted(() => ({
    context: (name: string) => ({ name, value: {} as Record<string, unknown> }),
    hasIncomingConnections: false,
}));

vi.mock('react', async (importOriginal) => ({
    ...(await importOriginal<typeof import('react')>()),
    ...(await import('./hookRunner')).fakeReact,
}));
vi.mock('use-context-selector', () => ({
    useContext: (context: { value: unknown }) => context.value,
    useContextSelector: (context: { value: unknown }, select: (value: unknown) => unknown) =>
        select(context.value),
}));
vi.mock('../../src/renderer/contexts/GlobalNodeState', () => ({
    GlobalContext: stubs.context('global'),
    GlobalVolatileContext: stubs.context('volatile'),
}));
vi.mock('../../src/renderer/contexts/AlertBoxContext', () => ({
    AlertBoxContext: stubs.context('alerts'),
}));
vi.mock('../../src/renderer/contexts/BackendContext', () => ({
    BackendContext: stubs.context('backend'),
}));
vi.mock('../../src/renderer/hooks/useSettings', () => ({
    useSettings: () => ({ packageSettings: {} }),
}));
vi.mock('../../src/renderer/hooks/useAutomaticFeatures', () => ({
    useAutomaticFeatures: () => ({
        isAutomatic: !stubs.hasIncomingConnections,
        hasIncomingConnections: stubs.hasIncomingConnections,
    }),
}));

const SCHEMA_ID = 'test:run:load' as SchemaId;
const schema = {
    schemaId: SCHEMA_ID,
    inputs: [{ id: 0 as InputId }],
};
const runIndividual = vi.fn<[{ inputs: unknown[] }], Promise<unknown>>();
const add = vi.fn();
const remove = vi.fn();
const sendToast = vi.fn();

const deferred = <T>() => {
    let resolve: (value: T) => void = () => {};
    let reject: (reason: unknown) => void = () => {};
    const promise = new Promise<T>((res, rej) => {
        resolve = res;
        reject = rej;
    });
    return { promise, resolve, reject };
};

const nodeData = (path: string): NodeData => ({
    id: 'node',
    schemaId: SCHEMA_ID,
    inputData: { [0 as InputId]: path },
});

// the hook gives other effects 50 ms to settle before it runs the node
const settle = () => vi.advanceTimersByTimeAsync(60);

beforeEach(async () => {
    vi.useFakeTimers();
    vi.clearAllMocks();
    stubs.hasIncomingConnections = false;
    const { GlobalContext, GlobalVolatileContext } = (await import(
        '../../src/renderer/contexts/GlobalNodeState'
    )) as unknown as Record<string, { value: unknown }>;
    GlobalContext.value = { addIndividuallyRunning: add, removeIndividuallyRunning: remove };
    GlobalVolatileContext.value = { chainLoads: 0 };
    const { AlertBoxContext } = (await import(
        '../../src/renderer/contexts/AlertBoxContext'
    )) as unknown as Record<string, { value: unknown }>;
    AlertBoxContext.value = { sendToast };
    const { BackendContext } = (await import(
        '../../src/renderer/contexts/BackendContext'
    )) as unknown as Record<string, { value: unknown }>;
    BackendContext.value = {
        schemata: { get: () => schema },
        backend: { runIndividual, clearNodeCacheIndividual: vi.fn().mockResolvedValue(undefined) },
        isBackendReady: true,
        isNodeLoaded: () => true,
    };
});

afterEach(() => {
    vi.useRealTimers();
});

test('a preview whose request fails does not stay running', async () => {
    // Stop aborts the requests of the backend, and a lost connection fails them
    const reply = deferred<unknown>();
    runIndividual.mockReturnValueOnce(reply.promise);
    mount(() => useRunNode(nodeData('a.png'), true));
    await settle();
    expect(add).toHaveBeenCalledWith('node');
    expect(remove).not.toHaveBeenCalled();

    reply.reject(new Error('connection lost'));
    await settle();

    expect(remove).toHaveBeenCalledWith('node');
});

test('a node runs when its last connection is removed with inputs it did not run with', async () => {
    runIndividual.mockResolvedValue({ success: true });
    stubs.hasIncomingConnections = true;
    let data = nodeData('a.png');
    const view = mount(() => useRunNode(data, true));
    await settle();
    expect(runIndividual).not.toHaveBeenCalled();

    stubs.hasIncomingConnections = false;
    view.render();
    await settle();
    expect(runIndividual).toHaveBeenCalledOnce();

    // connected for a while with another value
    stubs.hasIncomingConnections = true;
    view.render();
    await settle();
    data = nodeData('b.png');
    view.render();
    await settle();
    expect(runIndividual).toHaveBeenCalledOnce();

    stubs.hasIncomingConnections = false;
    view.render();
    await settle();
    expect(runIndividual).toHaveBeenCalledTimes(2);
    expect(runIndividual.mock.calls[1][0]).toMatchObject({ inputs: ['b.png'] });

    // nothing changed since that run
    stubs.hasIncomingConnections = true;
    view.render();
    await settle();
    stubs.hasIncomingConnections = false;
    view.render();
    await settle();
    expect(runIndividual).toHaveBeenCalledTimes(2);
});

test('a node stays running until the last of its overlapping previews has ended', async () => {
    const first = deferred<unknown>();
    const second = deferred<unknown>();
    runIndividual.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    let data = nodeData('a.png');
    const view = mount(() => useRunNode(data, true));
    await settle();
    data = nodeData('b.png');
    view.render();
    await settle();
    expect(runIndividual).toHaveBeenCalledTimes(2);

    first.resolve({ success: true });
    await settle();
    expect(remove).not.toHaveBeenCalled();

    second.resolve({ success: true });
    await settle();
    expect(remove).toHaveBeenCalledOnce();
});
