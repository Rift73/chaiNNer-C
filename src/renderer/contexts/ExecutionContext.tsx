import { evaluate } from '@chainner/navi';
import i18n from 'i18next';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from 'react-query';
import { useReactFlow } from 'reactflow';
import { createContext, useContext, useContextSelector } from 'use-context-selector';
import {
    Backend,
    BackendEventMap,
    BackendStatusResponse,
    BackendWorkerStatusResponse,
    CancelError,
} from '../../common/Backend';
import { EdgeData, InputData, NodeData, OutputId } from '../../common/common-types';
import { formatExecutionErrorMessage } from '../../common/formatExecutionErrorMessage';
import {
    RunPart,
    withClipViewer,
    withInputOverrides,
    withOnlyDownstreamOf,
    withOnlyUpTo,
} from '../../common/input-override-common';
import { log } from '../../common/log';
import { checkFeatures } from '../../common/nodes/checkFeatures';
import { checkNodeValidity } from '../../common/nodes/checkNodeValidity';
import { getConnectedInputs } from '../../common/nodes/connectedInputs';
import { optimizeChain } from '../../common/nodes/optimize';
import { toBackendJson } from '../../common/nodes/toBackendJson';
import { getChainnerScope } from '../../common/types/chainner-scope';
import { fromJson } from '../../common/types/json';
import { EMPTY_MAP, EMPTY_SET, assertNever, groupBy, mapInputValues } from '../../common/util';
import { bothValid } from '../../common/Validity';
import {
    ChainProgress,
    getInitialChainProgress,
    getTotalProgress,
    isOwnProgress,
    withNodeFinished,
    withNodeProgress,
} from '../helpers/chainProgress';
import {
    BackendEventSource,
    useBackendEventSource,
    useBackendEventSourceListener,
} from '../hooks/useBackendEventSource';
import { EventBacklog, useEventBacklog } from '../hooks/useEventBacklog';
import { computeInputHashes } from '../hooks/useInputHashes';
import { useMemoObject } from '../hooks/useMemo';
import { OutputDataEntry } from '../hooks/useOutputDataStore';
import { useSettings } from '../hooks/useSettings';
import { ipcRenderer } from '../safeIpc';
import { AlertBoxContext, AlertType } from './AlertBoxContext';
import { BackendContext } from './BackendContext';
import { GlobalContext, GlobalVolatileContext } from './GlobalNodeState';

// eslint-disable-next-line react-refresh/only-export-components
export enum ExecutionStatus {
    READY,
    RUNNING,
    PAUSED,
    KILLING,
}

// eslint-disable-next-line react-refresh/only-export-components
export enum NodeExecutionStatus {
    /**
     * The node has not been run yet and is awaiting execution.
     */
    YET_TO_RUN,
    /**
     * The node is currently running.
     */
    RUNNING,
    /**
     * The node has finished running successfully.
     */
    FINISHED,
    /**
     * The is not part of the current execution. It has not and will not be executed.
     */
    NOT_EXECUTING,
}

interface ExecutionStatusContextValue {
    status: ExecutionStatus;
    paused: boolean;
}

export interface NodeProgress {
    progress: number;
    eta: number;
    index: number;
    total: number;
}

export type PreviewRender =
    // backendTime: the node's own time in the backend (s), when the render tells it; sizes: the
    // preview widths it asked for (renderAlone's view sizes), if any
    | {
          readonly kind: 'rendered';
          readonly backendTime?: number;
          readonly sizes?: readonly number[];
      }
    | { readonly kind: 'failed'; readonly message: string }
    | { readonly kind: 'skipped' };

interface ExecutionContextValue {
    run: () => Promise<void>;
    /**
     * Runs a part of the chain as `run` does (`part`: a node and what it depends on, or what it
     * feeds with what that depends on), with the given input values of the given nodes (by node id) for this run
     * only: the chain in the editor, which is saved, and the nodes' input hashes keep their
     * values. Does nothing unless the executor is ready.
     */
    runWithOverrides: (overrides: ReadonlyMap<string, InputData>, part: RunPart) => Promise<void>;
    /**
     * Renders what the node shows (its Preview at its own Seek) with the given input values of
     * the given nodes (by node id): only the node and the nodes it depends on run, and the node
     * keeps what it shows although the overriding values change its items; the other overridden
     * nodes keep their outputs, as in a run with overrides. The render is no run of the chain:
     * the Run buttons and the execution number stay as they are, a failure is returned, not
     * alerted, and a Run asked for meanwhile starts once the render ends. A render asked for
     * during another starts once that one ends. Skipped while a run goes on or waits, also when
     * a run is asked for while the render waits. `withViewSizes` (a gesture's frame): the run
     * brings only the previews of the node's views on screen (setViewWidth).
     */
    renderPreview: (
        nodeId: string,
        overrides: ReadonlyMap<string, InputData>,
        withViewSizes?: boolean
    ) => Promise<PreviewRender>;
    /**
     * Renders the Preview of a node without connections by itself (/run/individual), with the
     * given values of its own inputs (by input id) for this render only: the editor's node keeps
     * its values. The frame is stored under the input hash of the values it was rendered with,
     * as soon as its event comes, and the render ends then (or with its failure, or skipped when
     * no frame came: a newer run of the node replaced it). One render of a node at a time is the
     * caller's (useSeekPreview's). Skipped while a run goes on or waits.
     */
    renderAlone: (
        nodeId: string,
        inputs: InputData,
        withViewSizes: boolean
    ) => Promise<PreviewRender>;
    /**
     * The width (px, as the view picks its preview by) of a view of the node's preview on screen
     * (`view`: which), or undefined when it is not on screen. A render by itself with view sizes
     * (renderAlone during a gesture) asks only for the previews of these views; while a maximized
     * one (`isMaximized`) is open, only for those: it covers the node.
     */
    setViewWidth: (
        nodeId: string,
        view: string,
        width: number | undefined,
        isMaximized?: boolean
    ) => void;
    /**
     * A gesture's end sets these values of the node's own inputs: their commit runs nothing
     * when the renders by themselves (renderAlone) have their frame, rendered or in flight, or
     * one of them will render it (`willRender`). Nothing is reserved while a run goes on or
     * waits. A reserved frame that ends without being shown has the node run again
     * (registerRunAgain).
     */
    reserveAlone: (nodeId: string, inputs: InputData, willRender: boolean) => void;
    /**
     * Whether the commit of these input values (useRunNode's JSON of them) has its frame
     * reserved (reserveAlone): the reservation is used then, once.
     */
    takeReserved: (nodeId: string, inputsKey: string) => boolean;
    /** A run of the node is sent: its reservation is no frame of its inputs any more. */
    forgetReserved: (nodeId: string) => void;
    /** How the node runs again (useRunNode's reload); returns its removal. */
    registerRunAgain: (nodeId: string, runAgain: () => void) => () => void;
    /**
     * Whether the node's picture is a frame of the renders by itself that follow its seek bar
     * (renderAlone) while its inputs are those the renders started from: one is in flight, or
     * the frame shown is the last one asked for. It is not outdated then.
     */
    isSeekFrame: (
        nodeId: string,
        inputHash: string | undefined,
        shownHash: string | undefined
    ) => boolean;
    /**
     * Shows the clip of the given output of the node in the viewer, as a Preview Clips node fed by
     * it shows it, without adding one to the chain: only the node and the nodes it depends on run.
     * The viewer it opens for the node reloads at the node's next preview. Does nothing unless the
     * executor is ready.
     */
    previewClip: (nodeId: string, outputId: OutputId) => Promise<void>;
    pause: () => Promise<void>;
    kill: () => Promise<void>;
    status: ExecutionStatus;
    paused: boolean;
    getNodeProgress: (nodeId: string) => NodeProgress | undefined;
    getNodeStatus: (nodeId: string) => NodeExecutionStatus;
    executionNumber: number;
}

export const ExecutionStatusContext = createContext<Readonly<ExecutionStatusContextValue>>({
    status: ExecutionStatus.READY,
    paused: false,
});

export const ExecutionContext = createContext<Readonly<ExecutionContextValue>>(
    {} as ExecutionContextValue
);

interface BackloggedEvent<K extends keyof BackendEventMap> {
    type: K;
    data: BackendEventMap[K];
}
type NodeEvents =
    | BackloggedEvent<'node-start'>
    | BackloggedEvent<'node-progress'>
    | BackloggedEvent<'node-broadcast'>
    | BackloggedEvent<'node-finish'>;

const useRegisterNodeEvents = (
    eventSource: BackendEventSource | null,
    backlog: EventBacklog<NodeEvents>,
    // a render's own event (renderAlone's), handled at once instead of with the backlog
    takeOwn: (event: NodeEvents) => boolean
) => {
    useBackendEventSourceListener(
        eventSource,
        'node-start',
        (f) => f && backlog.push({ type: 'node-start', data: f })
    );
    useBackendEventSourceListener(
        eventSource,
        'node-progress',
        (f) => f && backlog.push({ type: 'node-progress', data: f })
    );
    useBackendEventSourceListener(eventSource, 'node-broadcast', (f) => {
        if (!f) return;
        const event = { type: 'node-broadcast', data: f } as const;
        if (!takeOwn(event)) backlog.push(event);
    });
    useBackendEventSourceListener(eventSource, 'node-finish', (f) => {
        if (!f) return;
        const event = { type: 'node-finish', data: f } as const;
        if (!takeOwn(event)) backlog.push(event);
    });
};

// The renderer's session in the tokens of its renders by themselves: a late event of an earlier
// session (before a reload of the window) is none of this one's.
const SESSION = crypto.randomUUID();

/** A render of a node by itself (renderAlone) in flight. */
interface AloneRender {
    /** The input hash of the values it renders. */
    readonly hash: string;
    /** The preview widths it asked for, if any. */
    readonly sizes: readonly number[] | undefined;
    executionTime: number | undefined;
    settle: (render: PreviewRender) => void;
}
/** What the renders of a node by itself last asked for (renderAlone). */
interface SeekFrame {
    /** The input hash of the frame last asked for. */
    readonly hash: string;
    /** The node's input hash when it was asked for. */
    readonly base: string;
    readonly inFlight: boolean;
}

const useOnBackendStatus = (
    backend: Backend,
    statusChangesRef: React.MutableRefObject<number>,
    onChange: (data: BackendStatusResponse) => void
) => {
    const statusQuery = useQuery({
        queryKey: ['status', backend.url],
        queryFn: async () => {
            const statusChanges = statusChangesRef.current;
            return { ...(await backend.status()), statusChanges, timestamp: Date.now() };
        },
        cacheTime: 0,
        retry: 25,
        refetchOnWindowFocus: true,
        refetchInterval: 3000,
    });

    const lastDataRef = useRef<BackendStatusResponse | null>(null);

    useEffect(() => {
        if (statusQuery.data && statusQuery.data !== lastDataRef.current) {
            lastDataRef.current = statusQuery.data;
            // A reply asked for before the status last changed here tells the status before that.
            if (statusQuery.data.statusChanges === statusChangesRef.current) {
                onChange(statusQuery.data);
            }
        }
    }, [statusQuery.data, onChange, statusChangesRef]);
};

// eslint-disable-next-line @typescript-eslint/ban-types
export const ExecutionProvider = memo(({ children }: React.PropsWithChildren<{}>) => {
    const {
        typeStateRef,
        chainLineageRef,
        outputDataActions,
        getOutputData,
        getInputHash,
        setManualOutputType,
        clearManualTypes,
        flushInputChanges,
    } = useContext(GlobalContext);
    const {
        schemata,
        url,
        backend,
        ownsBackend,
        backendDownRef,
        isBackendReady,
        features,
        featureStates,
        categories,
        passthrough,
    } = useContext(BackendContext);
    const { packageSettings } = useSettings();

    const { sendAlert, sendToast } = useContext(AlertBoxContext);
    const nodeChanges = useContextSelector(GlobalVolatileContext, (c) => c.nodeChanges);
    const edgeChanges = useContextSelector(GlobalVolatileContext, (c) => c.edgeChanges);

    const { getNodes, getEdges } = useReactFlow<NodeData, EdgeData>();

    const [status, setStatus] = useState(ExecutionStatus.READY);
    const statusChangesRef = useRef(0);
    const setLocalStatus = useCallback((newStatus: ExecutionStatus) => {
        statusChangesRef.current += 1;
        setStatus(newStatus);
    }, []);
    // for the event handlers, which do not run with a render
    const statusRef = useRef(status);
    statusRef.current = status;
    const runsInFlightRef = useRef(0);
    // The last Preview render asked for (renderPreview), settled when it ends, after those asked
    // for before it; whether a run waits for it; and the failure of the render in flight as the
    // error event told it, which raises no alert.
    const previewRef = useRef<Promise<void>>();
    const runAskedRef = useRef(false);
    const previewFailureRef = useRef<string>();
    // Error events of renders and runs that ended before their event came.
    const lateFailuresRef = useRef(0);
    // The run of the chain this window started last, while its request is in flight: whether its
    // error event came, and the alert it tells unless the user stopped the run.
    const chainRunRef = useRef<{
        errorCame: boolean;
        error?: { message: string; trace: string };
    }>();
    // Per node, the last render by itself started (renderAlone): the JSON of its input values,
    // whether it is in flight, whether it showed its frame, and whether with every preview (not
    // only those of the views on screen).
    const lastAloneRef = useRef(
        new Map<string, { key: string; inFlight: boolean; rendered: boolean; complete: boolean }>()
    );
    // Per node, the input values (JSON) a gesture's end set whose frame is reserved
    // (reserveAlone), and whether their commit has used it.
    const reservedRef = useRef(new Map<string, { key: string; used: boolean }>());
    const runAgainRef = useRef(new Map<string, () => void>());

    const [executionNumber, setExecutionNumber] = useState(1);

    const [chainProgress, setChainProgress] = useState<ChainProgress>(EMPTY_MAP);
    const totalChainProgress = useMemo(() => getTotalProgress(chainProgress), [chainProgress]);

    const [nodeProgress, setNodeProgress] = useState<Record<string, NodeProgress | undefined>>({});
    const setNodeProgressImpl = useCallback(
        (nodeId: string, progress: NodeProgress) => {
            setNodeProgress((prev) => ({ ...prev, [nodeId]: progress }));
        },
        [setNodeProgress]
    );
    const getNodeProgress = useCallback((nodeId: string) => nodeProgress[nodeId], [nodeProgress]);

    type ActiveStatus = Exclude<NodeExecutionStatus, NodeExecutionStatus.NOT_EXECUTING>;
    const [nodeStatusMap, setNodeStatusMap] =
        useState<ReadonlyMap<string, ActiveStatus>>(EMPTY_MAP);
    const clearNodeStatusMap = useCallback(() => setNodeStatusMap(EMPTY_MAP), []);
    const setNodeStatus = useCallback((executionStatus: ActiveStatus, nodes: Iterable<string>) => {
        setNodeStatusMap((prev) => {
            const newMap = new Map(prev);
            for (const nodeId of nodes) {
                newMap.set(nodeId, executionStatus);
            }
            return newMap;
        });
    }, []);
    const getNodeStatus = useCallback(
        (nodeId: string) => nodeStatusMap.get(nodeId) ?? NodeExecutionStatus.NOT_EXECUTING,
        [nodeStatusMap]
    );

    useOnBackendStatus(backend, statusChangesRef, (data) => {
        if (data.worker && !('error' in data.worker)) {
            const executorStatus = data.worker.executor;
            // A run this window started ends with the reply to its request. A status reply can say
            // ready before that: asked after the run started, it can reach the worker first.
            if (executorStatus === 'ready' && runsInFlightRef.current > 0) return;
            // a Preview render is no run of the chain
            if (previewRef.current !== undefined) return;
            const statusMapping = {
                running: ExecutionStatus.RUNNING,
                paused: ExecutionStatus.PAUSED,
                killing: ExecutionStatus.KILLING,
                ready: ExecutionStatus.READY,
            } satisfies Record<BackendWorkerStatusResponse['executor'], ExecutionStatus>;
            setStatus(statusMapping[executorStatus]);
        }
    });

    useEffect(() => {
        if (status === ExecutionStatus.RUNNING || status === ExecutionStatus.PAUSED) {
            ipcRenderer.send('set-progress-bar', totalChainProgress);
        } else {
            ipcRenderer.send('set-progress-bar', null);
        }
    }, [status, totalChainProgress]);

    useEffect(() => {
        if (status !== ExecutionStatus.READY) {
            ipcRenderer.send('start-sleep-blocker');
        } else {
            ipcRenderer.send('stop-sleep-blocker');
            setChainProgress(EMPTY_MAP);
            setNodeProgress({});
            clearNodeStatusMap();
        }
    }, [status, clearNodeStatusMap]);

    const executingIteratorNodesRef = useRef<ReadonlySet<string>>(EMPTY_SET);
    // The nodes whose inputs the running chain overrides (runWithOverrides).
    const overriddenNodesRef = useRef<ReadonlySet<string>>(EMPTY_SET);
    // The node whose Preview the running chain renders (renderPreview), or whose clip it shows
    // (previewClip).
    const shownNodeRef = useRef<string>();
    // The nodes shown by a gesture's renders that bring only the previews of their views on
    // screen. A broadcast marked as such a render's (its previewSizes) is a gesture frame of one
    // of them: any other node's is left out, so that no node keeps a picture without its other
    // previews; the gesture's end renders them all.
    const reducedShownRef = useRef(new Set<string>());
    // Per node shown by a gesture's renders, what it showed with every preview before them
    // (undefined: nothing): a failed end of the gesture shows it again (outdated), not a frame
    // without its previews.
    const beforeReducedRef = useRef(new Map<string, OutputDataEntry | undefined>());

    const processSingleNode = (nodeId: string, events: NodeEvents[]) => {
        let executionStatus;
        let executionTime;
        let broadcastData;
        let types;
        let progress;
        let chainNodeProgress;
        let sequenceTypes;
        // whether the broadcast stored is a gesture's frame with only some previews
        let isReduced = false;

        for (const { type, data } of events) {
            if (type === 'node-start') {
                executionStatus = NodeExecutionStatus.RUNNING as const;
            } else if (type === 'node-finish') {
                executionStatus = NodeExecutionStatus.FINISHED as const;
                executionTime = data.executionTime;
                chainNodeProgress = 1;
            } else if (type === 'node-broadcast') {
                // An overridden node's broadcast follows from the overriding values (Run this
                // frame: a range of one frame), which its own inputs do not have, and would be
                // kept under its own inputs' hash. It is left out: the node keeps the outputs
                // and types of its own inputs.
                const marked = data.previewSizes !== undefined;
                if (marked && !reducedShownRef.current.has(nodeId)) {
                    // a gesture's render with only the shown node's previews, whenever it comes
                } else if (!overriddenNodesRef.current.has(nodeId)) {
                    isReduced = marked;
                    broadcastData = data.data;
                    types = data.types;
                    sequenceTypes = data.sequenceTypes ?? undefined;
                } else if (nodeId === shownNodeRef.current) {
                    isReduced = marked;
                    // The run renders what the node shows for its own Seek, Frame Count and FPS:
                    // only the length of its frames is the overriding range's.
                    broadcastData = data.data;
                    types = data.types;
                }
            } else {
                progress = data;
                // A node's own progress stays out of the chain bar: the bar would move with each
                // tile of a long upscale and fall back at its next pass.
                if (!isOwnProgress(data)) {
                    chainNodeProgress = data.progress;
                }
            }
        }

        // Events can come after their run ended: the reply to the run comes over another
        // connection, and an error is handled ahead of the events before it. What they say of the
        // node's outputs holds, but the node is not running anymore.
        const executing = statusRef.current !== ExecutionStatus.READY;

        if (executing && executionStatus !== undefined) {
            setNodeStatus(executionStatus, [nodeId]);
        }

        if (
            executionTime !== undefined ||
            broadcastData !== undefined ||
            types !== undefined ||
            sequenceTypes !== undefined
        ) {
            // TODO: This is incorrect. The inputs of the node might have changed since
            // the chain started running. However, sending the then current input hashes
            // of the chain to the backend along with the rest of its data and then making
            // the backend send us those hashes is incorrect too because of iterators, I
            // think.
            const inputHash = getInputHash(nodeId);
            if (broadcastData !== undefined) {
                if (!isReduced) beforeReducedRef.current.delete(nodeId);
                else if (!beforeReducedRef.current.has(nodeId)) {
                    beforeReducedRef.current.set(nodeId, getOutputData(nodeId));
                }
            }
            outputDataActions.set(
                nodeId,
                executionTime,
                inputHash,
                broadcastData,
                types,
                sequenceTypes
            );
        }

        if (executing && progress) {
            setNodeProgressImpl(nodeId, progress);
        }

        if (executing && executionStatus === NodeExecutionStatus.FINISHED) {
            setNodeProgress((prev) => withNodeFinished(prev, nodeId));
        }

        if (chainNodeProgress !== undefined) {
            const p = chainNodeProgress;
            setChainProgress((prev) => withNodeProgress(prev, nodeId, p));
        }

        if (executing && executingIteratorNodesRef.current.has(nodeId) && types) {
            // we want to update the output types of the iterator nodes
            for (const [outputId, typeExpr] of Object.entries(types)) {
                if (typeExpr) {
                    try {
                        const type = evaluate(fromJson(typeExpr), getChainnerScope());
                        setManualOutputType(nodeId, Number(outputId) as OutputId, type);
                    } catch (error) {
                        log.error(error);
                    }
                }
            }
        }
    };
    const nodeEventBacklog = useEventBacklog({
        process: (events: NodeEvents[]) => {
            const byNodeId = groupBy(events, (e) => e.data.nodeId);
            for (const [nodeId, nodeEvents] of byNodeId) {
                processSingleNode(nodeId, nodeEvents);
            }
        },
        interval: 100,
    });

    // The renders of nodes by themselves in flight (renderAlone), by their requests' tokens.
    const aloneRendersRef = useRef(new Map<string, AloneRender>());
    const takeOwn = useCallback(
        (event: NodeEvents): boolean => {
            if (event.type !== 'node-broadcast' && event.type !== 'node-finish') return false;
            const { token } = event.data;
            if (token === undefined) return false;
            const render = aloneRendersRef.current.get(token);
            // the frame of a render that has ended is no frame the node shows
            if (render === undefined) return true;
            if (event.type === 'node-finish') {
                render.executionTime = event.data.executionTime;
                return true;
            }
            const { nodeId, data, types, sequenceTypes } = event.data;
            if (!getNodes().some((n) => n.id === nodeId)) {
                // the node is gone: its frame is no output of a node
                render.settle({ kind: 'skipped' });
                return true;
            }
            // a broadcast of the node waiting in the backlog is older than this frame
            nodeEventBacklog.discard(
                (e) => e.type === 'node-broadcast' && e.data.nodeId === nodeId
            );
            outputDataActions.set(
                nodeId,
                render.executionTime,
                render.hash,
                data,
                types,
                sequenceTypes ?? undefined
            );
            render.settle({
                kind: 'rendered',
                backendTime: render.executionTime,
                sizes: render.sizes,
            });
            return true;
        },
        [outputDataActions, getNodes, nodeEventBacklog]
    );

    const [eventSource, eventSourceStatus] = useBackendEventSource(url, isBackendReady);
    useRegisterNodeEvents(eventSource, nodeEventBacklog, takeOwn);

    const formatError = useCallback(
        (error: Pick<BackendEventMap['execution-error'], 'exception' | 'source'>) =>
            formatExecutionErrorMessage(
                error,
                schemata,
                (label, value) => `• ${label}: ${value}`,
                (nodeId) => getNodes().find((n) => n.id === nodeId)?.data.nodeName
            ),
        [schemata, getNodes]
    );

    useBackendEventSourceListener(eventSource, 'execution-error', (data) => {
        if (data) {
            const formatted = formatError(data);
            // The backend sends the errors of its runs in their order: while a render that ended
            // before its event came waits for it, the next error is that render's, not the one
            // of a render in flight.
            if (lateFailuresRef.current > 0) {
                lateFailuresRef.current -= 1;
                return;
            }
            // A Preview render's failure goes to the render, which reports it once for a drag.
            if (previewRef.current !== undefined) {
                previewFailureRef.current = formatted;
                return;
            }
            // A run this window started is alerted by its reply, which says all the event does: the
            // event comes before the reply, after it, or not at all (lost with a broken stream).
            const run = chainRunRef.current;
            if (run !== undefined) run.errorCame = true;
            if (statusRef.current === ExecutionStatus.KILLING) {
                // Stop is still ending this run, and the user asked for it to end.
                log.warn(`A stopped run failed: ${data.exception}`);
                return;
            }
            if (run !== undefined) {
                // for a reply that does not come (its request cancelled by a restart)
                run.error = { message: formatted, trace: data.exceptionTrace };
            } else {
                // a run this window did not start (one started before a reload of the window)
                sendAlert({
                    type: AlertType.ERROR,
                    message: formatted,
                    trace: data.exceptionTrace,
                });
            }
            clearNodeStatusMap();
            setLocalStatus(ExecutionStatus.READY);
        }
    });

    useBackendEventSourceListener(eventSource, 'execution-warning', (data) => {
        if (data) {
            // The run goes on; the toast stays until it is closed.
            sendToast({ status: 'warning', description: data.message, duration: null });
        }
    });

    useBackendEventSourceListener(eventSource, 'chain-start', (data) => {
        // The backend sends the events of its runs in their order: an earlier run's error event
        // has come by now, or never will.
        lateFailuresRef.current = 0;
        if (data && statusRef.current !== ExecutionStatus.READY) {
            clearNodeStatusMap();
            setNodeStatus(NodeExecutionStatus.YET_TO_RUN, data.nodes);

            // the backend might have optimized away some nodes and we have to
            // take that into account for progress
            setChainProgress((prev) => {
                const newProgress = new Map(prev);
                const allowed = new Set(data.nodes);
                for (const key of [...newProgress.keys()].filter((k) => !allowed.has(k))) {
                    newProgress.delete(key);
                }
                return newProgress;
            });
        }
    });

    useEffect(() => {
        if (ownsBackend && !backendDownRef.current && eventSourceStatus === 'error') {
            log.warn('The backend event source errored.');
            // Their frames may not come (the event in flight when the stream broke is lost), and
            // a reply that says its frame was sent waits for it: the first one's failure is
            // alerted (once, by its seek bar), the others end without a frame.
            [...aloneRendersRef.current.values()].forEach((render, index) =>
                render.settle(
                    index === 0
                        ? {
                              kind: 'failed',
                              message: i18n.t(
                                  'execution.unexpectedError',
                                  'An unexpected error occurred.'
                              ),
                          }
                        : { kind: 'skipped' }
                )
            );
            // The stream reconnects by itself and the backend keeps the events made meanwhile: a
            // run this window started goes on, and ends with the reply to its request (also when
            // the backend goes down).
            if (runsInFlightRef.current === 0) {
                clearNodeStatusMap();
                setLocalStatus(ExecutionStatus.READY);
            }
        }
    }, [eventSourceStatus, clearNodeStatusMap, backendDownRef, ownsBackend, setLocalStatus]);

    const lastChangesRef = useRef(`${nodeChanges} ${edgeChanges}`);
    useEffect(() => {
        const currentChanges = `${nodeChanges} ${edgeChanges}`;
        if (lastChangesRef.current === currentChanges) return;
        lastChangesRef.current = currentChanges;

        // a run that renders a node's Preview follows the changes that asked for it
        if (status === ExecutionStatus.RUNNING && shownNodeRef.current === undefined) {
            sendToast({
                status: 'warning',
                description:
                    'You are modifying the chain while it is running. This will not modify the state of the current execution.',
                id: 'execution-running',
                variant: 'subtle',
                position: 'bottom',
            });
        } else if (status === ExecutionStatus.PAUSED) {
            sendToast({
                status: 'warning',
                description:
                    'You are modifying the chain while it is paused. This will not modify the state of the execution once resumed.',
                id: 'execution-paused',
                variant: 'subtle',
                position: 'bottom',
            });
        }
    }, [status, nodeChanges, edgeChanges, sendToast]);

    // per node, the widths of the views of its preview on screen, by view
    const viewWidthsRef = useRef(
        new Map<string, Map<string, { width: number; isMaximized: boolean }>>()
    );
    const setViewWidth = useCallback(
        (nodeId: string, view: string, width: number | undefined, isMaximized = false) => {
            const widths =
                viewWidthsRef.current.get(nodeId) ??
                new Map<string, { width: number; isMaximized: boolean }>();
            if (width === undefined) widths.delete(view);
            else widths.set(view, { width, isMaximized });
            if (widths.size > 0) viewWidthsRef.current.set(nodeId, widths);
            else viewWidthsRef.current.delete(nodeId);
        },
        []
    );
    // The preview widths the views of the node on screen pick, none when none is known (then
    // every preview): an open maximized view covers the node, its picture alone is seen.
    const viewSizesOf = useCallback((nodeId: string): number[] | undefined => {
        const views = [...(viewWidthsRef.current.get(nodeId)?.values() ?? [])];
        const seen = views.some((v) => v.isMaximized) ? views.filter((v) => v.isMaximized) : views;
        if (seen.length === 0) return undefined;
        return [...new Set(seen.map((v) => Math.ceil(v.width)))].sort((a, b) => a - b);
    }, []);
    const runNodes = useCallback(
        async (
            overrides: ReadonlyMap<string, InputData>,
            part?: RunPart,
            shownNodeId?: string,
            clipOutput?: OutputId,
            previewSizes?: readonly number[]
        ): Promise<string | undefined> => {
            // A Preview render (a shown node, no viewer) leaves the Run buttons alone and returns
            // its failure instead of alerting it.
            const isPreview = shownNodeId !== undefined && clipOutput === undefined;
            let failure: string | undefined;
            const fail = (message: string) => {
                failure ??= message;
                if (!isPreview) sendAlert({ type: AlertType.ERROR, message });
            };
            flushInputChanges();
            // The overriding values go into copies of the nodes: the editor's nodes and their
            // input hashes keep theirs.
            const overridden = withInputOverrides(getNodes(), getEdges(), overrides);
            let chain = overridden;
            if (part?.kind === 'up-to') {
                chain = withOnlyUpTo(overridden.nodes, overridden.edges, part.nodeId);
            } else if (part?.kind === 'downstream') {
                chain = withOnlyDownstreamOf(overridden.nodes, overridden.edges, part.nodeId);
            }
            if (shownNodeId !== undefined && clipOutput !== undefined) {
                chain = withClipViewer(chain.nodes, chain.edges, shownNodeId, clipOutput);
            }
            const { nodes, edges, report } = optimizeChain(
                chain.nodes,
                chain.edges,
                schemata,
                passthrough
            );

            // show an error if there are no nodes to run
            if (nodes.length === 0) {
                let message;
                if (report.removedSideEffectFree > 0) {
                    message = i18n.t(
                        'execution.noEffectiveNodes',
                        'There are no nodes that have an effect. Try to view or output images/files.'
                    );
                } else if (report.removedDisabled > 0) {
                    message = i18n.t(
                        'execution.allNodesDisabled',
                        'All nodes are disabled. There are no nodes to run.'
                    );
                } else {
                    message = i18n.t('execution.noNodes', 'There are no nodes to run.');
                }
                fail(message);
                return failure;
            }

            // check for static errors
            const invalidNodes = nodes.flatMap((node) => {
                // The editor's types describe a node's own values, not the overriding ones (a
                // saved range of Start after End is invalid, the overriding one-frame range is
                // not): an overridden node is checked without them, and the backend checks the
                // values it runs.
                const functionInstance = overrides.has(node.data.id)
                    ? undefined
                    : typeStateRef.current.functions.get(node.data.id);
                const schema = schemata.get(node.data.schemaId);
                const { name } = schema;
                const category = categories.get(schema.category)?.name ?? schema.category;

                const validity = bothValid(
                    checkFeatures(schema.features, features, featureStates),
                    checkNodeValidity({
                        inputData: node.data.inputData,
                        connectedInputs: getConnectedInputs(node.id, edges),
                        schema,
                        functionInstance,
                        chainLineage: chainLineageRef.current,
                        nodeId: node.id,
                    })
                );
                if (validity.isValid) return [];

                return [`• ${category}: ${name}: ${validity.reason}`];
            });
            if (invalidNodes.length > 0) {
                const reasons = invalidNodes.join('\n');
                fail(
                    `${i18n.t(
                        'execution.invalidNodes',
                        'There are invalid nodes in the editor. Please fix them before running.'
                    )}\n${reasons}`
                );
                return failure;
            }

            // find iterator nodes for later
            const iteratorNodeIds = new Set(
                nodes
                    .filter((n) => schemata.get(n.data.schemaId).kind === 'generator')
                    .map((n) => n.data.id)
            );
            if (!isPreview) {
                executingIteratorNodesRef.current = iteratorNodeIds;
                setChainProgress(getInitialChainProgress(nodes, schemata));
            }
            overriddenNodesRef.current = new Set(overrides.keys());
            shownNodeRef.current = shownNodeId;
            if (previewSizes !== undefined && shownNodeId !== undefined) {
                reducedShownRef.current.add(shownNodeId);
            }

            const run: NonNullable<typeof chainRunRef.current> = { errorCame: false };
            runsInFlightRef.current += 1;
            try {
                if (!isPreview) {
                    chainRunRef.current = run;
                    setLocalStatus(ExecutionStatus.RUNNING);
                }

                const data = toBackendJson(nodes, edges, schemata);
                const response = await backend.run({
                    data,
                    options: packageSettings,
                    sendBroadcastData: true,
                    ...(previewSizes === undefined ? {} : { previewSizes: [...previewSizes] }),
                });
                if (response.type === 'already-running') {
                    fail(
                        i18n.t(
                            'execution.previousStillRunning',
                            'Cannot start because a previous executor is still running.'
                        )
                    );
                } else if (response.type === 'error' && isPreview) {
                    // its event may come after the reply: it is the render's too
                    if (previewFailureRef.current === undefined) lateFailuresRef.current += 1;
                    failure ??= previewFailureRef.current ?? response.exception;
                } else if (response.type === 'error') {
                    // The reply reports the failure, with its trace; its event, if it comes
                    // after the reply, is dropped. A run the user stopped reports none.
                    if (!run.errorCame) lateFailuresRef.current += 1;
                    if (statusRef.current !== ExecutionStatus.KILLING) {
                        const { exception, source = null, exceptionTrace } = response;
                        const message =
                            formatError({ exception, source }) ||
                            i18n.t('execution.unexpectedError', 'An unexpected error occurred.');
                        failure ??= message;
                        sendAlert({ type: AlertType.ERROR, message, trace: exceptionTrace });
                    }
                }
            } catch (err: unknown) {
                if (!(err instanceof CancelError)) {
                    fail(
                        i18n.t(
                            'execution.unexpectedErrorDetail',
                            'An unexpected error occurred: {{error}}',
                            { error: String(err) }
                        )
                    );
                } else if (run.error !== undefined && backendDownRef.current) {
                    // A restart of the backend cancelled the request of a run that had failed
                    // (Stop cancels it with the backend up): its reply will not tell, its event does.
                    failure ??= run.error.message;
                    sendAlert({ type: AlertType.ERROR, ...run.error });
                }
            } finally {
                runsInFlightRef.current -= 1;
                // A run started after this one (once this one's error event ended its display)
                // keeps its own display.
                const isCurrent = chainRunRef.current === run;
                if (isCurrent) chainRunRef.current = undefined;
                nodeEventBacklog.processAll();
                overriddenNodesRef.current = EMPTY_SET;
                shownNodeRef.current = undefined;
                if (!isPreview) {
                    if (isCurrent) {
                        clearNodeStatusMap();
                        setLocalStatus(ExecutionStatus.READY);
                    }
                    clearManualTypes(iteratorNodeIds);
                }
            }
            return failure;
        },
        [
            flushInputChanges,
            getNodes,
            getEdges,
            schemata,
            categories,
            sendAlert,
            typeStateRef,
            passthrough,
            chainLineageRef,
            features,
            featureStates,
            backend,
            packageSettings,
            clearNodeStatusMap,
            nodeEventBacklog,
            clearManualTypes,
            setLocalStatus,
            formatError,
            backendDownRef,
        ]
    );

    const resume = useCallback(async () => {
        try {
            const response = await backend.resume();
            if (response.type === 'error') {
                sendAlert({ type: AlertType.ERROR, message: response.exception });
                return;
            }
            if (response.type === 'no-executor') {
                return;
            }
            setLocalStatus(ExecutionStatus.RUNNING);
        } catch (err) {
            sendAlert({
                type: AlertType.ERROR,
                message: `${i18n.t(
                    'execution.unexpectedError',
                    'An unexpected error occurred.'
                )}\n\n${String(err)}`,
            });
        }
    }, [backend, sendAlert, setLocalStatus]);

    // A run waits for the Preview render in flight, and no render starts while it waits.
    const afterPreview = useCallback(async (start: () => Promise<unknown>) => {
        runAskedRef.current = true;
        try {
            await previewRef.current;
            await start();
        } finally {
            runAskedRef.current = false;
        }
    }, []);

    const run = useCallback(async () => {
        if (!isBackendReady) {
            // the backend does not take requests yet (Run is disabled until it does)
            return;
        }
        if (status === ExecutionStatus.PAUSED) {
            await resume();
        } else {
            await afterPreview(() => runNodes(EMPTY_MAP));
            setExecutionNumber((prev) => prev + 1);
        }
    }, [isBackendReady, resume, runNodes, status, setExecutionNumber, afterPreview]);

    const runWithOverrides = useCallback(
        async (overrides: ReadonlyMap<string, InputData>, part: RunPart) => {
            if (!isBackendReady || status !== ExecutionStatus.READY) {
                // a paused run is not resumed with other values, and a run is not started twice
                return;
            }
            await afterPreview(() => runNodes(overrides, part));
            setExecutionNumber((prev) => prev + 1);
        },
        [isBackendReady, runNodes, status, setExecutionNumber, afterPreview]
    );

    const renderPreview = useCallback(
        async (
            nodeId: string,
            overrides: ReadonlyMap<string, InputData>,
            withViewSizes = false
        ): Promise<PreviewRender> => {
            // a run goes on or waits to start
            const isRunAhead = () =>
                runAskedRef.current || statusRef.current !== ExecutionStatus.READY;
            if (!isBackendReady || status !== ExecutionStatus.READY || isRunAhead()) {
                return { kind: 'skipped' };
            }
            // Renders follow one another in the order asked: this one waits for the one before
            // it, and the next one for this one.
            const before = previewRef.current;
            let settle = () => {};
            const rendering = new Promise<void>((resolve) => {
                settle = resolve;
            });
            previewRef.current = rendering;
            try {
                await before;
                if (isRunAhead()) return { kind: 'skipped' };
                previewFailureRef.current = undefined;
                const sizes = withViewSizes ? viewSizesOf(nodeId) : undefined;
                const failure = await runNodes(
                    overrides,
                    { nodeId, kind: 'up-to' },
                    nodeId,
                    undefined,
                    sizes
                );
                if (
                    failure !== undefined &&
                    sizes === undefined &&
                    beforeReducedRef.current.has(nodeId)
                ) {
                    // The end of a gesture failed: the node shows its last complete picture,
                    // or none when it had none (a run renders it then).
                    const shownBefore = beforeReducedRef.current.get(nodeId);
                    beforeReducedRef.current.delete(nodeId);
                    if (shownBefore === undefined) {
                        outputDataActions.delete(nodeId);
                    } else {
                        outputDataActions.set(
                            nodeId,
                            shownBefore.lastExecutionTime,
                            shownBefore.inputHash,
                            shownBefore.data,
                            shownBefore.types,
                            shownBefore.sequenceTypes
                        );
                    }
                }
                return failure === undefined
                    ? { kind: 'rendered', sizes }
                    : { kind: 'failed', message: failure };
            } finally {
                if (previewRef.current === rendering) previewRef.current = undefined;
                settle();
            }
        },
        [isBackendReady, runNodes, status, viewSizesOf, outputDataActions]
    );

    const tokensRef = useRef(0);
    const [seekFrames, setSeekFrames] = useState<ReadonlyMap<string, SeekFrame>>(EMPTY_MAP);
    // The node by itself with the given values of its inputs, unless no render can start.
    const aloneWith = useCallback(
        (nodeId: string, inputs: InputData) => {
            const isRunAhead = runAskedRef.current || statusRef.current !== ExecutionStatus.READY;
            const node = getNodes().find((n) => n.id === nodeId);
            if (!isBackendReady || isRunAhead || node === undefined) return undefined;
            const schema = schemata.get(node.data.schemaId);
            const data = { ...node.data, inputData: { ...node.data.inputData, ...inputs } };
            const values = mapInputValues(schema, (inputId) => data.inputData[inputId] ?? null);
            // useRunNode's key of the values
            const key = JSON.stringify(values);
            // the node has no connections: its hash is its own values'
            const hash = computeInputHashes([{ ...node, data }], [], schemata).get(nodeId) ?? '';
            return { schemaId: node.data.schemaId, values, key, hash };
        },
        [isBackendReady, getNodes, schemata]
    );
    const reserveAlone = useCallback(
        (nodeId: string, inputs: InputData, willRender: boolean) => {
            const alone = aloneWith(nodeId, inputs);
            if (alone === undefined) return;
            const last = lastAloneRef.current.get(nodeId);
            // a frame with only some previews is no frame at rest: the end's render is
            const hasFrame =
                last?.key === alone.key && last.complete && (last.inFlight || last.rendered);
            if (!hasFrame && !willRender) return;
            reservedRef.current.set(nodeId, { key: alone.key, used: false });
            // the frames go on following the bar, whose value is now these inputs'
            setSeekFrames((prev) => {
                const frame = prev.get(nodeId);
                if (frame === undefined) return prev;
                return new Map(prev).set(nodeId, { ...frame, base: alone.hash });
            });
        },
        [aloneWith]
    );
    // the reservation ends; a commit that used it runs the node again
    const endReserved = useCallback((nodeId: string) => {
        const reserved = reservedRef.current.get(nodeId);
        reservedRef.current.delete(nodeId);
        if (reserved?.used) runAgainRef.current.get(nodeId)?.();
    }, []);
    const takeReserved = useCallback((nodeId: string, inputsKey: string) => {
        const reserved = reservedRef.current.get(nodeId);
        if (reserved?.key !== inputsKey || reserved.used) return false;
        const last = lastAloneRef.current.get(nodeId);
        if (last?.key === inputsKey && !last.inFlight && last.rendered && last.complete) {
            // its frame is shown with every preview: nothing more to wait for
            reservedRef.current.delete(nodeId);
        } else {
            reservedRef.current.set(nodeId, { ...reserved, used: true });
        }
        return true;
    }, []);
    const forgetReserved = useCallback((nodeId: string) => {
        reservedRef.current.delete(nodeId);
    }, []);
    const registerRunAgain = useCallback((nodeId: string, runAgain: () => void) => {
        runAgainRef.current.set(nodeId, runAgain);
        return () => {
            if (runAgainRef.current.get(nodeId) === runAgain) runAgainRef.current.delete(nodeId);
        };
    }, []);
    const renderAlone = useCallback(
        async (
            nodeId: string,
            inputs: InputData,
            withViewSizes: boolean
        ): Promise<PreviewRender> => {
            const alone = aloneWith(nodeId, inputs);
            if (alone === undefined) {
                // a run goes on or waits: a reserved frame waited for comes from the node's run
                if (reservedRef.current.has(nodeId)) endReserved(nodeId);
                return { kind: 'skipped' };
            }
            const { schemaId, values, key, hash } = alone;
            const base = getInputHash(nodeId);
            tokensRef.current += 1;
            const token = `${nodeId} ${SESSION} ${tokensRef.current}`;
            // only the previews the views on screen pick
            const sizes = withViewSizes ? viewSizesOf(nodeId) : undefined;
            const render: AloneRender = {
                hash,
                sizes,
                executionTime: undefined,
                settle: () => {},
            };
            const done = new Promise<PreviewRender>((resolve) => {
                render.settle = resolve;
            });
            aloneRendersRef.current.set(token, render);
            const complete = sizes === undefined;
            lastAloneRef.current.set(nodeId, { key, inFlight: true, rendered: false, complete });
            // another frame of the node: its next gesture, past the reserved end
            if (reservedRef.current.get(nodeId)?.key !== key) reservedRef.current.delete(nodeId);
            setSeekFrames((prev) => new Map(prev).set(nodeId, { hash, base, inFlight: true }));
            // The frame's event ends the render, also before the reply (which comes once the
            // frame is sent); a reply without a frame or a failure ends it too.
            backend
                .runIndividual({
                    schemaId,
                    id: nodeId,
                    inputs: values,
                    options: packageSettings,
                    token,
                    ...(sizes === undefined ? {} : { previewSizes: sizes }),
                })
                .then(
                    (reply) => {
                        if (!reply.success) {
                            render.settle({ kind: 'failed', message: reply.error });
                        } else if (!reply.data?.broadcast) {
                            // a newer run of the node stopped this one: no frame comes
                            render.settle({ kind: 'skipped' });
                        }
                    },
                    (error: unknown) => {
                        render.settle({ kind: 'failed', message: String(error) });
                    }
                )
                .catch(log.error);
            const result = await done;
            aloneRendersRef.current.delete(token);
            const rendered = result.kind === 'rendered';
            if (lastAloneRef.current.get(nodeId)?.key === key) {
                lastAloneRef.current.set(nodeId, { key, inFlight: false, rendered, complete });
            }
            const reserved = reservedRef.current.get(nodeId);
            if (reserved?.key === key) {
                // A render with only the views' previews leaves the reservation to the end's
                // render with every one, which follows unless the render failed.
                if (complete ? !rendered : result.kind === 'failed') endReserved(nodeId);
                else if (complete && reserved.used) reservedRef.current.delete(nodeId);
            } else if (reserved !== undefined && result.kind === 'failed') {
                // after a failure the seek bar renders nothing more: the reserved frame neither
                endReserved(nodeId);
            }
            setSeekFrames((prev) => {
                const frame = prev.get(nodeId);
                if (frame?.hash !== hash || !frame.inFlight) return prev;
                return new Map(prev).set(nodeId, { ...frame, inFlight: false });
            });
            return result;
        },
        [aloneWith, getInputHash, backend, packageSettings, endReserved, viewSizesOf]
    );
    // A node that leaves the chain leaves its renders by itself: their frames are dropped.
    useEffect(() => {
        const ids = new Set(getNodes().map((n) => n.id));
        for (const nodeId of [
            ...lastAloneRef.current.keys(),
            ...reservedRef.current.keys(),
            ...beforeReducedRef.current.keys(),
            ...reducedShownRef.current,
        ]) {
            if (!ids.has(nodeId)) {
                lastAloneRef.current.delete(nodeId);
                reservedRef.current.delete(nodeId);
                beforeReducedRef.current.delete(nodeId);
                reducedShownRef.current.delete(nodeId);
            }
        }
        setSeekFrames((prev) => {
            const gone = [...prev.keys()].filter((nodeId) => !ids.has(nodeId));
            if (gone.length === 0) return prev;
            const kept = new Map(prev);
            for (const nodeId of gone) kept.delete(nodeId);
            return kept;
        });
    }, [nodeChanges, getNodes]);
    const isSeekFrame = useCallback(
        (nodeId: string, inputHash: string | undefined, shownHash: string | undefined) => {
            const frame = seekFrames.get(nodeId);
            return (
                frame !== undefined &&
                frame.base === inputHash &&
                (frame.inFlight || frame.hash === shownHash)
            );
        },
        [seekFrames]
    );

    const previewClip = useCallback(
        async (nodeId: string, outputId: OutputId) => {
            if (!isBackendReady || status !== ExecutionStatus.READY) return;
            await afterPreview(() =>
                runNodes(EMPTY_MAP, { nodeId, kind: 'up-to' }, nodeId, outputId)
            );
        },
        [isBackendReady, runNodes, status, afterPreview]
    );

    const pause = useCallback(async () => {
        try {
            const response = await backend.pause();
            if (response.type === 'error') {
                sendAlert({ type: AlertType.ERROR, message: response.exception });
                return;
            }
            if (response.type === 'no-executor') {
                return;
            }
            setLocalStatus(ExecutionStatus.PAUSED);
        } catch (err) {
            sendAlert({
                type: AlertType.ERROR,
                message: `${i18n.t(
                    'execution.unexpectedError',
                    'An unexpected error occurred.'
                )}\n\n${String(err)}`,
            });
        }
    }, [backend, sendAlert, setLocalStatus]);

    const kill = useCallback(async () => {
        try {
            setLocalStatus(ExecutionStatus.KILLING);
            backend.abort();
            // We need to set the status again, since run() resets it to READY
            // We have to do this in a setTimeout, otherwise react doesn't honor this
            const killBackendAndWait = new Promise<void>((resolve, reject) =>
                // eslint-disable-next-line no-promise-executor-return, @typescript-eslint/no-misused-promises
                setTimeout(async () => {
                    setLocalStatus(ExecutionStatus.KILLING);
                    try {
                        const response = await backend.kill();
                        // The host's /kill returns once a restarted worker answers, and a
                        // run waits for the nodes it needs, so the node list is not fetched.
                        if (response.type === 'error') {
                            sendAlert({ type: AlertType.ERROR, message: response.exception });
                        }
                    } catch (err) {
                        reject(err);
                    } finally {
                        setLocalStatus(ExecutionStatus.READY);
                        resolve();
                    }
                }, 0)
            );
            await killBackendAndWait;
        } catch (err) {
            sendAlert({
                type: AlertType.ERROR,
                message: `${i18n.t(
                    'execution.unexpectedError',
                    'An unexpected error occurred.'
                )}\n\n${String(err)}`,
            });
        }
        setNodeProgress({});
    }, [backend, sendAlert, setLocalStatus]);

    // This makes sure keystrokes are executed even if the focus is on an input field
    useEffect(() => {
        const handleKeyPress = (event: KeyboardEvent) => {
            if (event.code === 'F5') {
                switch (status) {
                    case ExecutionStatus.READY:
                    case ExecutionStatus.PAUSED:
                        event.preventDefault();
                        run().catch(log.error);
                        break;
                    case ExecutionStatus.RUNNING:
                    case ExecutionStatus.KILLING:
                        break;
                    default:
                        assertNever(status);
                }
            } else if (event.code === 'F6') {
                switch (status) {
                    case ExecutionStatus.RUNNING:
                        event.preventDefault();
                        pause().catch(log.error);
                        break;
                    case ExecutionStatus.READY:
                    case ExecutionStatus.PAUSED:
                    case ExecutionStatus.KILLING:
                        break;
                    default:
                        assertNever(status);
                }
            } else if (event.code === 'F7') {
                switch (status) {
                    case ExecutionStatus.RUNNING:
                    case ExecutionStatus.PAUSED:
                        event.preventDefault();
                        kill().catch(log.error);
                        break;
                    case ExecutionStatus.READY:
                    case ExecutionStatus.KILLING:
                        break;
                    default:
                        assertNever(status);
                }
            }
        };

        document.addEventListener('keydown', handleKeyPress);

        return () => {
            document.removeEventListener('keydown', handleKeyPress);
        };
    }, [run, pause, kill, status]);

    const statusValue = useMemoObject<ExecutionStatusContextValue>({
        status,
        paused: status === ExecutionStatus.PAUSED,
    });

    const value = useMemoObject<ExecutionContextValue>({
        run,
        runWithOverrides,
        renderPreview,
        renderAlone,
        setViewWidth,
        reserveAlone,
        takeReserved,
        forgetReserved,
        registerRunAgain,
        isSeekFrame,
        previewClip,
        pause,
        kill,
        status,
        paused: status === ExecutionStatus.PAUSED,
        getNodeProgress,
        getNodeStatus,
        executionNumber,
    });

    return (
        <ExecutionContext.Provider value={value}>
            <ExecutionStatusContext.Provider value={statusValue}>
                {children}
                <div style={{ display: 'none' }}>
                    {status};{totalChainProgress}
                </div>
            </ExecutionStatusContext.Provider>
        </ExecutionContext.Provider>
    );
});
