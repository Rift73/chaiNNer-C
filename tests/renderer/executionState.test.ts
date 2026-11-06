import { Node } from 'reactflow';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { BackendEventMap, CancelError } from '../../src/common/Backend';
import {
    CategoryId,
    NodeData,
    NodeGroupId,
    NodeSchema,
    SchemaId,
} from '../../src/common/common-types';
import { PassthroughMap } from '../../src/common/PassthroughMap';
import { SchemaMap } from '../../src/common/SchemaMap';
import {
    ExecutionProvider,
    ExecutionStatus,
    NodeExecutionStatus,
} from '../../src/renderer/contexts/ExecutionContext';
import { mount } from './hookRunner';

const stubs = vi.hoisted(() => ({
    context: (name: string) => ({ name, value: {} as Record<string, unknown>, Provider: name }),
    listeners: new Map<string, (data: unknown) => void>(),
    eventSource: { status: 'open' },
    query: {
        queryFn: undefined as undefined | (() => Promise<unknown>),
        data: undefined as unknown,
    },
    nodes: [] as unknown[],
    send: vi.fn<[string, unknown?], void>(),
}));

vi.mock('react', async (importOriginal) => ({
    ...(await importOriginal<typeof import('react')>()),
    ...(await import('./hookRunner')).fakeReact,
}));
vi.mock('use-context-selector', () => ({
    createContext: () => stubs.context('context'),
    useContext: (context: { value: unknown }) => context.value,
    useContextSelector: (context: { value: unknown }, select: (value: unknown) => unknown) =>
        select(context.value),
}));
vi.mock('react-query', () => ({
    useQuery: (options: { queryFn: () => Promise<unknown> }) => {
        stubs.query.queryFn = options.queryFn;
        return { data: stubs.query.data };
    },
}));
vi.mock('reactflow', () => ({
    useReactFlow: () => ({ getNodes: () => stubs.nodes, getEdges: () => [] }),
}));
vi.mock('../../src/renderer/contexts/GlobalNodeState', () => ({
    GlobalContext: stubs.context('global'),
    GlobalVolatileContext: stubs.context('volatile'),
}));
vi.mock('../../src/renderer/contexts/AlertBoxContext', () => ({
    AlertBoxContext: stubs.context('alerts'),
    AlertType: { ERROR: 'error' },
}));
vi.mock('../../src/renderer/contexts/BackendContext', () => ({
    BackendContext: stubs.context('backend'),
}));
vi.mock('../../src/renderer/hooks/useBackendEventSource', () => ({
    useBackendEventSource: () => [{}, stubs.eventSource.status],
    useBackendEventSourceListener: (
        _source: unknown,
        type: string,
        listener: (data: unknown) => void
    ) => {
        stubs.listeners.set(type, listener);
    },
}));
vi.mock('../../src/renderer/hooks/useSettings', () => ({
    useSettings: () => ({ packageSettings: {} }),
}));
vi.mock('../../src/renderer/safeIpc', () => ({ ipcRenderer: { send: stubs.send } }));

const schema: NodeSchema = {
    schemaId: 'test:execution:view' as SchemaId,
    name: 'View',
    category: 'test' as CategoryId,
    nodeGroup: 'test' as NodeGroupId,
    description: '',
    seeAlso: [],
    icon: '',
    kind: 'regularNode',
    inputs: [],
    outputs: [],
    groupLayout: [],
    iteratorInputs: [],
    iteratorOutputs: [],
    hasSideEffects: true,
    deprecated: false,
    features: [],
    suggestions: [],
};
const node = (id: string): Node<NodeData> => ({
    id,
    type: 'regularNode',
    position: { x: 0, y: 0 },
    data: { id, schemaId: schema.schemaId, inputData: {} },
});

interface ProviderElement {
    props: {
        value: {
            run: () => Promise<void>;
            kill: () => Promise<void>;
            status: ExecutionStatus;
            getNodeStatus: (id: string) => NodeExecutionStatus;
            getNodeProgress: (id: string) => unknown;
        };
    };
}

const deferred = <T>() => {
    let resolve: (value: T) => void = () => {};
    let reject: (reason: unknown) => void = () => {};
    const promise = new Promise<T>((res, rej) => {
        resolve = res;
        reject = rej;
    });
    return { promise, resolve, reject };
};

const backend = {
    url: 'http://backend',
    run: vi.fn(),
    kill: vi.fn(),
    abort: vi.fn(),
    status: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
};
const sendAlert = vi.fn();
const outputSet = vi.fn();

const start = () => {
    const view = mount(() => ExecutionProvider({}) as unknown as ProviderElement);
    const execution = () => view.result.props.value;
    const emit = <K extends keyof BackendEventMap>(type: K, data: BackendEventMap[K]) => {
        stubs.listeners.get(type)!(data);
    };
    // the node events are looked at every 100 ms
    const tick = () => vi.advanceTimersByTimeAsync(100);
    return { view, execution, emit, tick };
};
const progress = (nodeId: string, value: number): BackendEventMap['node-progress'] => ({
    nodeId,
    progress: value,
    index: 1,
    total: 2,
    eta: 1,
});

const contexts = async () => {
    const { GlobalContext, GlobalVolatileContext } = (await import(
        '../../src/renderer/contexts/GlobalNodeState'
    )) as unknown as Record<string, { value: unknown }>;
    const { AlertBoxContext } = (await import(
        '../../src/renderer/contexts/AlertBoxContext'
    )) as unknown as Record<string, { value: unknown }>;
    const { BackendContext } = (await import(
        '../../src/renderer/contexts/BackendContext'
    )) as unknown as Record<string, { value: unknown }>;
    return { GlobalContext, GlobalVolatileContext, AlertBoxContext, BackendContext };
};

beforeEach(async () => {
    vi.useFakeTimers();
    vi.stubGlobal('document', { addEventListener: vi.fn(), removeEventListener: vi.fn() });
    vi.clearAllMocks();
    stubs.eventSource.status = 'open';
    stubs.query.data = undefined;
    stubs.nodes = [node('a')];
    const { GlobalContext, GlobalVolatileContext, AlertBoxContext, BackendContext } =
        await contexts();
    GlobalContext.value = {
        typeStateRef: { current: { functions: new Map() } },
        chainLineageRef: { current: undefined },
        outputDataActions: { set: outputSet },
        getInputHash: () => 'hash',
        setManualOutputType: vi.fn(),
        clearManualTypes: vi.fn(),
        flushInputChanges: vi.fn(),
    };
    GlobalVolatileContext.value = { nodeChanges: 0, edgeChanges: 0 };
    AlertBoxContext.value = { sendAlert, sendToast: vi.fn() };
    BackendContext.value = {
        schemata: new SchemaMap([schema]),
        url: backend.url,
        backend,
        ownsBackend: true,
        backendDownRef: { current: false },
        isBackendReady: true,
        features: new Map(),
        featureStates: new Map(),
        categories: { get: () => undefined },
        passthrough: PassthroughMap.create(new Map()),
    };
});

afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
});

const startRun = async (context: ReturnType<typeof start>) => {
    const reply = deferred<unknown>();
    backend.run.mockReturnValueOnce(reply.promise);
    const running = context.execution().run();
    await vi.advanceTimersByTimeAsync(0);
    expect(context.execution().status).toBe(ExecutionStatus.RUNNING);
    return { reply, running };
};

test('an event that arrives after its run has ended leaves no node running', async () => {
    const context = start();
    const { reply, running } = await startRun(context);

    // the reply to the run and the events come over different connections
    reply.resolve({ type: 'success' });
    await running;
    await vi.advanceTimersByTimeAsync(0);
    expect(context.execution().status).toBe(ExecutionStatus.READY);

    context.emit('chain-start', { nodes: ['a'] });
    await context.tick();
    expect(context.execution().getNodeStatus('a')).toBe(NodeExecutionStatus.NOT_EXECUTING);

    context.emit('node-start', { nodeId: 'a' } as BackendEventMap['node-start']);
    context.emit('node-progress', progress('a', 0.5));
    await context.tick();
    expect(context.execution().getNodeStatus('a')).toBe(NodeExecutionStatus.NOT_EXECUTING);
    expect(context.execution().getNodeProgress('a')).toBeUndefined();

    // what it reports still belongs to the node
    context.emit('node-finish', { nodeId: 'a', executionTime: 7 });
    await context.tick();
    expect(context.execution().getNodeStatus('a')).toBe(NodeExecutionStatus.NOT_EXECUTING);
    expect(outputSet).toHaveBeenCalledWith('a', 7, 'hash', undefined, undefined, undefined);
});

test('a broadcast after the end of a run does not narrow the types of an iterator', async () => {
    const generator = { ...schema, schemaId: 'test:execution:load' as SchemaId, kind: 'generator' };
    const { BackendContext, GlobalContext } = await contexts();
    (BackendContext.value as Record<string, unknown>).schemata = new SchemaMap([
        schema,
        generator as NodeSchema,
    ]);
    const setManualOutputType = vi.fn();
    (GlobalContext.value as Record<string, unknown>).setManualOutputType = setManualOutputType;
    stubs.nodes = [
        {
            ...node('load'),
            type: 'generator',
            data: { id: 'load', schemaId: generator.schemaId, inputData: {} },
        },
    ];
    const broadcast = {
        nodeId: 'load',
        data: {},
        types: { 0: 'uint' },
    } as unknown as BackendEventMap['node-broadcast'];
    const context = start();
    const { reply, running } = await startRun(context);

    context.emit('node-broadcast', broadcast);
    await context.tick();
    expect(setManualOutputType).toHaveBeenCalledOnce();

    reply.resolve({ type: 'success' });
    await running;
    await vi.advanceTimersByTimeAsync(0);

    // the last event of the run, or the preview of the node, whose types no run end clears
    context.emit('node-broadcast', broadcast);
    await context.tick();
    expect(setManualOutputType).toHaveBeenCalledOnce();
});

test('an iterator progress event just before an error shows no progress afterwards', async () => {
    const context = start();
    const { reply, running } = await startRun(context);

    // the error is handled at once, the progress waits for the next look at the node events
    context.emit('node-progress', progress('a', 0.5));
    context.emit('execution-error', {
        message: 'Error running nodes!',
        source: null,
        exception: 'failed',
        exceptionTrace: '',
    });
    await vi.advanceTimersByTimeAsync(0);
    await context.tick();
    reply.resolve({ type: 'error' });
    await running;
    await context.tick();

    expect(context.execution().getNodeProgress('a')).toBeUndefined();
});

test('a node that finishes in one batch with its progress counts as done', async () => {
    stubs.nodes = [node('a'), node('b')];
    const context = start();
    const { reply, running } = await startRun(context);

    context.emit('node-progress', progress('a', 0.25));
    context.emit('node-finish', { nodeId: 'a', executionTime: 1 });
    await context.tick();

    const bar = stubs.send.mock.calls.filter(([channel]) => channel === 'set-progress-bar');
    expect(bar[bar.length - 1][1]).toBe(0.5);

    // and a progress event after a finish is the state of a node's next iteration
    context.emit('node-finish', { nodeId: 'b', executionTime: 1 });
    context.emit('node-progress', progress('b', 0.5));
    await context.tick();
    const later = stubs.send.mock.calls.filter(([channel]) => channel === 'set-progress-bar');
    expect(later[later.length - 1][1]).toBe(0.75);

    reply.resolve({ type: 'success' });
    await running;
});

test("a node's own progress stays out of the chain bar and ends at 100% with its finish", async () => {
    stubs.nodes = [node('a'), node('b')];
    const context = start();
    const { reply, running } = await startRun(context);
    const lastBar = () => {
        const bar = stubs.send.mock.calls.filter(([channel]) => channel === 'set-progress-bar');
        return bar[bar.length - 1][1];
    };
    const own = (nodeId: string, value: number): BackendEventMap['node-progress'] => ({
        nodeId,
        progress: value,
        index: 0,
        total: 0,
        eta: 0,
    });

    context.emit('node-progress', own('a', 0.37));
    await context.tick();
    expect(lastBar()).toBe(0);
    expect(context.execution().getNodeProgress('a')).toEqual(own('a', 0.37));

    // its finish counts whatever its last own progress said, also in one batch with it
    context.emit('node-progress', own('a', 0.75));
    context.emit('node-finish', { nodeId: 'a', executionTime: 1 });
    await context.tick();
    expect(lastBar()).toBe(0.5);
    expect(context.execution().getNodeProgress('a')).toEqual(own('a', 1));

    // the next iteration's own progress does not take the finish back in the chain bar
    context.emit('node-start', { nodeId: 'a' });
    context.emit('node-progress', own('a', 0.25));
    await context.tick();
    expect(lastBar()).toBe(0.5);
    expect(context.execution().getNodeProgress('a')).toEqual(own('a', 0.25));

    reply.resolve({ type: 'success' });
    await running;
});

test('an error after Stop is no alert and does not end the stopping', async () => {
    const context = start();
    const { reply, running } = await startRun(context);
    const killed = deferred<unknown>();
    backend.kill.mockReturnValueOnce(killed.promise);
    backend.abort.mockImplementationOnce(() => reply.reject(new CancelError('aborted')));

    const killing = context.execution().kill();
    await vi.advanceTimersByTimeAsync(0);
    await running;
    await vi.advanceTimersByTimeAsync(0);
    expect(context.execution().status).toBe(ExecutionStatus.KILLING);

    context.emit('execution-error', {
        message: 'Error running nodes!',
        source: null,
        exception: 'cancelled while it failed',
        exceptionTrace: '',
    });
    await vi.advanceTimersByTimeAsync(0);

    expect(sendAlert).not.toHaveBeenCalled();
    expect(context.execution().status).toBe(ExecutionStatus.KILLING);

    killed.resolve({ type: 'success' });
    await killing;
    await vi.advanceTimersByTimeAsync(0);
    expect(context.execution().status).toBe(ExecutionStatus.READY);
});

test('a failed run does not clear the node states of the next run a second later', async () => {
    const context = start();
    const failed = await startRun(context);
    failed.reply.reject(new Error('connection lost'));
    await failed.running;
    await vi.advanceTimersByTimeAsync(0);
    expect(sendAlert).toHaveBeenCalledOnce();

    const next = await startRun(context);
    context.emit('chain-start', { nodes: ['a'] });
    await vi.advanceTimersByTimeAsync(1100);
    expect(context.execution().getNodeStatus('a')).toBe(NodeExecutionStatus.YET_TO_RUN);

    next.reply.resolve({ type: 'success' });
    await next.running;
});

test('a status reply that reached the worker before the run does not end the run', async () => {
    const context = start();
    const { reply, running } = await startRun(context);
    const poll = async (executor: string) => {
        backend.status.mockResolvedValueOnce({ ready: true, worker: { executor } });
        stubs.query.data = await stubs.query.queryFn!();
        context.view.render();
        await vi.advanceTimersByTimeAsync(0);
    };

    // asked after the run started, with the request of the run still on its way
    await poll('ready');
    expect(context.execution().status).toBe(ExecutionStatus.RUNNING);

    context.emit('chain-start', { nodes: ['a'] });
    await vi.advanceTimersByTimeAsync(0);
    expect(context.execution().getNodeStatus('a')).toBe(NodeExecutionStatus.YET_TO_RUN);

    reply.resolve({ type: 'success' });
    await running;
    await vi.advanceTimersByTimeAsync(0);
    expect(context.execution().status).toBe(ExecutionStatus.READY);

    // a run this window did not start (it was reloaded): the replies are all it knows
    await poll('running');
    expect(context.execution().status).toBe(ExecutionStatus.RUNNING);
    await poll('ready');
    expect(context.execution().status).toBe(ExecutionStatus.READY);
});

test('a status reply from before a run started does not end the run', async () => {
    const context = start();
    const statusReply = deferred<unknown>();
    backend.status.mockReturnValueOnce(statusReply.promise);
    const polling = stubs.query.queryFn!();

    const { reply, running } = await startRun(context);
    context.emit('chain-start', { nodes: ['a'] });

    statusReply.resolve({ ready: true, worker: { executor: 'ready' } });
    stubs.query.data = await polling;
    context.view.render();
    await vi.advanceTimersByTimeAsync(0);

    expect(context.execution().status).toBe(ExecutionStatus.RUNNING);
    expect(context.execution().getNodeStatus('a')).toBe(NodeExecutionStatus.YET_TO_RUN);

    // a reply that was asked for after the run started counts
    backend.status.mockResolvedValueOnce({ ready: true, worker: { executor: 'paused' } });
    stubs.query.data = await stubs.query.queryFn!();
    context.view.render();
    await vi.advanceTimersByTimeAsync(0);
    expect(context.execution().status).toBe(ExecutionStatus.PAUSED);

    reply.resolve({ type: 'success' });
    await running;
});
