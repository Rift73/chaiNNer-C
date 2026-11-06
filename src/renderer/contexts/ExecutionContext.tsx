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
import { withInputOverrides } from '../../common/input-override-common';
import { log } from '../../common/log';
import { checkFeatures } from '../../common/nodes/checkFeatures';
import { checkNodeValidity } from '../../common/nodes/checkNodeValidity';
import { getConnectedInputs } from '../../common/nodes/connectedInputs';
import { optimizeChain } from '../../common/nodes/optimize';
import { toBackendJson } from '../../common/nodes/toBackendJson';
import { getChainnerScope } from '../../common/types/chainner-scope';
import { fromJson } from '../../common/types/json';
import { EMPTY_MAP, EMPTY_SET, assertNever, groupBy } from '../../common/util';
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
import { useMemoObject } from '../hooks/useMemo';
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

interface ExecutionContextValue {
    run: () => Promise<void>;
    /**
     * Runs the chain as `run` does, with the given input values of the given nodes (by node id)
     * for this run only: the chain in the editor, which is saved, and the nodes' input hashes
     * keep their values. Does nothing unless the executor is ready.
     */
    runWithOverrides: (overrides: ReadonlyMap<string, InputData>) => Promise<void>;
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
    backlog: EventBacklog<NodeEvents>
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
    useBackendEventSourceListener(
        eventSource,
        'node-broadcast',
        (f) => f && backlog.push({ type: 'node-broadcast', data: f })
    );
    useBackendEventSourceListener(
        eventSource,
        'node-finish',
        (f) => f && backlog.push({ type: 'node-finish', data: f })
    );
};

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

    const processSingleNode = (nodeId: string, events: NodeEvents[]) => {
        let executionStatus;
        let executionTime;
        let broadcastData;
        let types;
        let progress;
        let chainNodeProgress;
        let sequenceTypes;

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
                if (!overriddenNodesRef.current.has(nodeId)) {
                    broadcastData = data.data;
                    types = data.types;
                    sequenceTypes = data.sequenceTypes ?? undefined;
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

    const [eventSource, eventSourceStatus] = useBackendEventSource(url, isBackendReady);
    useRegisterNodeEvents(eventSource, nodeEventBacklog);

    useBackendEventSourceListener(eventSource, 'execution-error', (data) => {
        if (data) {
            if (statusRef.current === ExecutionStatus.KILLING) {
                // Stop is still ending this run, and the user asked for it to end.
                log.warn(`A stopped run failed: ${data.exception}`);
                return;
            }
            sendAlert({
                type: AlertType.ERROR,
                message: formatExecutionErrorMessage(
                    data,
                    schemata,
                    (label, value) => `• ${label}: ${value}`,
                    (nodeId) => getNodes().find((n) => n.id === nodeId)?.data.nodeName
                ),
                trace: data.exceptionTrace,
            });
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
            clearNodeStatusMap();
            setLocalStatus(ExecutionStatus.READY);
        }
    }, [eventSourceStatus, clearNodeStatusMap, backendDownRef, ownsBackend, setLocalStatus]);

    const lastChangesRef = useRef(`${nodeChanges} ${edgeChanges}`);
    useEffect(() => {
        const currentChanges = `${nodeChanges} ${edgeChanges}`;
        if (lastChangesRef.current === currentChanges) return;
        lastChangesRef.current = currentChanges;

        if (status === ExecutionStatus.RUNNING) {
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

    const runNodes = useCallback(
        async (overrides: ReadonlyMap<string, InputData>) => {
            flushInputChanges();
            // The overriding values go into copies of the nodes: the editor's nodes and their
            // input hashes keep theirs.
            const overridden = withInputOverrides(getNodes(), getEdges(), overrides);
            const { nodes, edges, report } = optimizeChain(
                overridden.nodes,
                overridden.edges,
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
                sendAlert({ type: AlertType.ERROR, message });
                return;
            }

            // check for static errors
            const invalidNodes = nodes.flatMap((node) => {
                // The editor's types describe a node's own values, not the overriding ones (a
                // saved range of In after Out is invalid, the overriding one-frame range is
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
                sendAlert({
                    type: AlertType.ERROR,
                    message: `${i18n.t(
                        'execution.invalidNodes',
                        'There are invalid nodes in the editor. Please fix them before running.'
                    )}\n${reasons}`,
                });
                return;
            }

            // find iterator nodes for later
            const iteratorNodeIds = new Set(
                nodes
                    .filter((n) => schemata.get(n.data.schemaId).kind === 'generator')
                    .map((n) => n.data.id)
            );
            executingIteratorNodesRef.current = iteratorNodeIds;
            overriddenNodesRef.current = new Set(overrides.keys());
            setChainProgress(getInitialChainProgress(nodes, edges, schemata));

            runsInFlightRef.current += 1;
            try {
                setLocalStatus(ExecutionStatus.RUNNING);

                const data = toBackendJson(nodes, edges, schemata);
                const response = await backend.run({
                    data,
                    options: packageSettings,
                    sendBroadcastData: true,
                });
                if (response.type === 'error') {
                    // no need to alert here, because the error has already been handled by the queue
                }
                if (response.type === 'already-running') {
                    sendAlert({
                        type: AlertType.ERROR,
                        message: i18n.t(
                            'execution.previousStillRunning',
                            'Cannot start because a previous executor is still running.'
                        ),
                    });
                }
            } catch (err: unknown) {
                if (!(err instanceof CancelError)) {
                    sendAlert({
                        type: AlertType.ERROR,
                        message: i18n.t(
                            'execution.unexpectedErrorDetail',
                            'An unexpected error occurred: {{error}}',
                            { error: String(err) }
                        ),
                    });
                }
            } finally {
                runsInFlightRef.current -= 1;
                nodeEventBacklog.processAll();
                overriddenNodesRef.current = EMPTY_SET;
                clearNodeStatusMap();
                setLocalStatus(ExecutionStatus.READY);
                clearManualTypes(iteratorNodeIds);
            }
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
                message: i18n.t('execution.unexpectedError', 'An unexpected error occurred.'),
            });
        }
    }, [backend, sendAlert, setLocalStatus]);

    const run = useCallback(async () => {
        if (!isBackendReady) {
            // the backend does not take requests yet (Run is disabled until it does)
            return;
        }
        if (status === ExecutionStatus.PAUSED) {
            await resume();
        } else {
            await runNodes(EMPTY_MAP);
            setExecutionNumber((prev) => prev + 1);
        }
    }, [isBackendReady, resume, runNodes, status, setExecutionNumber]);

    const runWithOverrides = useCallback(
        async (overrides: ReadonlyMap<string, InputData>) => {
            if (!isBackendReady || status !== ExecutionStatus.READY) {
                // a paused run is not resumed with other values, and a run is not started twice
                return;
            }
            await runNodes(overrides);
            setExecutionNumber((prev) => prev + 1);
        },
        [isBackendReady, runNodes, status, setExecutionNumber]
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
                message: i18n.t('execution.unexpectedError', 'An unexpected error occurred.'),
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
                message: i18n.t('execution.unexpectedError', 'An unexpected error occurred.'),
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
