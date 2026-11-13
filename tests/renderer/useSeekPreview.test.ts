import { beforeEach, expect, test, vi } from 'vitest';
import { useSeekPreview } from '../../src/renderer/hooks/useSeekPreview';
import { mount } from './hookRunner';

const stubs = vi.hoisted(() => ({
    context: (name: string) => ({ name, value: {} as Record<string, unknown> }),
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
vi.mock('../../src/renderer/contexts/AlertBoxContext', () => ({
    AlertBoxContext: stubs.context('alerts'),
    AlertType: { ERROR: 'error' },
}));
vi.mock('../../src/renderer/contexts/BackendContext', () => ({
    BackendContext: stubs.context('backend'),
}));
vi.mock('../../src/renderer/contexts/ExecutionContext', () => ({
    ExecutionContext: stubs.context('execution'),
    ExecutionStatus: { READY: 0, RUNNING: 1 },
}));

const READY = 0;
const RUNNING = 1;

type Render = { kind: 'rendered' } | { kind: 'failed'; message: string } | { kind: 'skipped' };

const deferred = () => {
    let resolve: (render: Render) => void = () => {};
    const promise = new Promise<Render>((res) => {
        resolve = res;
    });
    return { promise, resolve };
};

const renderPreview = vi.fn<[string, unknown], Promise<Render>>();
const sendAlert = vi.fn();
const overridesFor = (frame: number) => ({ 3: 'range', 4: frame, 5: frame });

const contexts = async () => {
    const { AlertBoxContext } = (await import(
        '../../src/renderer/contexts/AlertBoxContext'
    )) as unknown as Record<string, { value: unknown }>;
    const { BackendContext } = (await import(
        '../../src/renderer/contexts/BackendContext'
    )) as unknown as Record<string, { value: unknown }>;
    const { ExecutionContext } = (await import(
        '../../src/renderer/contexts/ExecutionContext'
    )) as unknown as Record<string, { value: Record<string, unknown> }>;
    return { AlertBoxContext, BackendContext, ExecutionContext };
};

const setStatus = async (status: number) => {
    const { ExecutionContext } = await contexts();
    ExecutionContext.value = { ...ExecutionContext.value, status };
};

/** Gives every render that is to start the time to start. */
const settle = () =>
    new Promise((resolve) => {
        setTimeout(resolve, 10);
    });

/** The frames rendered, in order. */
const rendered = () => renderPreview.mock.calls.map(([, overrides]) => overrides);

beforeEach(async () => {
    renderPreview.mockReset();
    sendAlert.mockReset();
    const { AlertBoxContext, BackendContext, ExecutionContext } = await contexts();
    AlertBoxContext.value = { sendAlert };
    BackendContext.value = { isBackendReady: true };
    ExecutionContext.value = { status: READY, renderPreview };
});

test('a frame renders through the nodes before the node, for that node', async () => {
    renderPreview.mockResolvedValue({ kind: 'rendered' });
    const view = mount(() => useSeekPreview('node', overridesFor));

    view.result(5, true);
    await vi.waitFor(() => expect(renderPreview).toHaveBeenCalledOnce());

    expect(renderPreview).toHaveBeenCalledWith('node', overridesFor(5));
});

test('during a render only the latest frame asked for renders after it', async () => {
    const first = deferred();
    renderPreview.mockReturnValueOnce(first.promise).mockResolvedValue({ kind: 'rendered' });
    const view = mount(() => useSeekPreview('node', overridesFor));

    view.result(5, true);
    await vi.waitFor(() => expect(renderPreview).toHaveBeenCalledOnce());

    view.result(6, false);
    view.result(7, false);
    view.result(8, false);
    await Promise.resolve();
    expect(rendered()).toEqual([overridesFor(5)]);

    first.resolve({ kind: 'rendered' });
    await vi.waitFor(() => expect(renderPreview).toHaveBeenCalledTimes(2));
    expect(rendered()).toEqual([overridesFor(5), overridesFor(8)]);

    // and nothing after it
    await settle();
    expect(renderPreview).toHaveBeenCalledTimes(2);
});

test('a run of the chain in progress is left alone: nothing renders, then or after it', async () => {
    renderPreview.mockResolvedValue({ kind: 'rendered' });
    await setStatus(RUNNING);
    const view = mount(() => useSeekPreview('node', overridesFor));

    view.result(5, true);
    await setStatus(READY);
    view.render();
    await settle();

    expect(renderPreview).not.toHaveBeenCalled();
});

test('a failing render is alerted once, and the rest of the drag renders nothing', async () => {
    const first = deferred();
    renderPreview.mockReturnValueOnce(first.promise).mockResolvedValue({
        kind: 'failed',
        message: 'the clip is gone',
    });
    const view = mount(() => useSeekPreview('node', overridesFor));

    view.result(5, true);
    await vi.waitFor(() => expect(renderPreview).toHaveBeenCalledOnce());
    view.result(6, false);
    first.resolve({ kind: 'failed', message: 'the clip is gone' });
    await settle();
    for (let frame = 7; frame < 30; frame += 1) {
        view.result(frame, false);
    }
    await settle();

    expect(rendered()).toEqual([overridesFor(5)]);
    expect(sendAlert).toHaveBeenCalledOnce();
    expect(sendAlert).toHaveBeenCalledWith({ type: 'error', message: 'the clip is gone' });

    // the next drag tries again
    view.result(40, true);
    await vi.waitFor(() => expect(renderPreview).toHaveBeenCalledTimes(2));
    expect(rendered()).toEqual([overridesFor(5), overridesFor(40)]);
});
