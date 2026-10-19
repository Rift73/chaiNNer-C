import { Scope } from '@chainner/navi';
import isDeepEqual from 'fast-deep-equal';
import React, {
    MutableRefObject,
    memo,
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
} from 'react';
import { useTranslation } from 'react-i18next';
import { useQuery, useQueryClient } from 'react-query';
import { createContext, useContext } from 'use-context-selector';
import { Backend, BackendData, BackendEventMap, getBackend } from '../../common/Backend';
import { CategoryMap } from '../../common/CategoryMap';
import {
    CategoryId,
    Feature,
    FeatureId,
    FeatureState,
    Package,
    PythonInfo,
    SchemaId,
} from '../../common/common-types';
import { log } from '../../common/log';
import { parseFunctionDefinitions } from '../../common/nodes/parseFunctionDefinitions';
import { sortNodes } from '../../common/nodes/sort';
import { PassthroughMap } from '../../common/PassthroughMap';
import { SchemaInputsMap } from '../../common/SchemaInputsMap';
import { SchemaMap } from '../../common/SchemaMap';
import { getChainnerScope } from '../../common/types/chainner-scope';
import { FunctionDefinition } from '../../common/types/function';
import { EMPTY_ARRAY, EMPTY_MAP, EMPTY_SET, delay } from '../../common/util';
import { useAsyncEffect } from '../hooks/useAsyncEffect';
import { BackendEventSource } from '../hooks/useBackendEventSource';
import { useMemoObject } from '../hooks/useMemo';
import { ipcRenderer } from '../safeIpc';
import { AlertBoxContext, AlertId, AlertType } from './AlertBoxContext';

interface BackendContextState {
    url: string;
    backend: Backend;
    ownsBackend: boolean;
    schemata: SchemaMap;
    categories: CategoryMap;
    categoriesMissingNodes: readonly CategoryId[];
    schemaInputs: SchemaInputsMap;
    pythonInfo: PythonInfo;
    packages: readonly Package[];
    passthrough: PassthroughMap;
    functionDefinitions: ReadonlyMap<SchemaId, FunctionDefinition>;
    scope: Scope;
    features: ReadonlyMap<FeatureId, Feature>;
    featureStates: ReadonlyMap<FeatureId, FeatureState>;
    refreshFeatureStates: () => Promise<void>;
    backendDownRef: React.MutableRefObject<boolean>;
    restart: () => Promise<void>;
    connectionState: 'connecting' | 'connected' | 'failed';
    /**
     * Whether the backend accepts requests. Runs it cannot serve yet (their nodes are still
     * loading) wait there.
     */
    isBackendReady: boolean;
    /**
     * Whether the node list is the copy cached at the last start, which the backend's own list
     * replaces once it arrives (if it differs).
     */
    nodesFromCache: boolean;
    /**
     * Whether the backend has loaded the node of this schema. A request that needs a node that
     * still loads waits for it in the backend, which holds one of the window's 6 connections to
     * the backend, so previews wait for this instead.
     */
    isNodeLoaded: (schemaId: SchemaId) => boolean;
    /** The backend's setup events, e.g. the progress of dependency installs. */
    setupEventSource: BackendEventSource | null;
    refreshNodes: () => void;
}

export const BackendContext = createContext<Readonly<BackendContextState>>(
    {} as BackendContextState
);

interface BackendProviderProps {
    url: string;
    pythonInfo: PythonInfo;
    /** The node list cached at the last start, if there is a current one. */
    cachedData: BackendData | undefined;
}

interface NodesInfo {
    rawResponse: BackendData;
    schemata: SchemaMap;
    categories: CategoryMap;
    passthrough: PassthroughMap;
    functionDefinitions: ReadonlyMap<SchemaId, FunctionDefinition>;
    categoriesMissingNodes: CategoryId[];
    packages: Package[];
    /** Whether this is the list cached at the last start, not yet confirmed by the backend. */
    fromCache: boolean;
}

const processBackendResponse = (rawResponse: BackendData, fromCache = false): NodesInfo => {
    const { categories, categoriesMissingNodes, nodes } = rawResponse[0];
    const categoryMap = new CategoryMap(categories);
    const schemata = new SchemaMap(sortNodes(nodes, categoryMap));

    const functionDefinitions = parseFunctionDefinitions(nodes);
    const passthrough = PassthroughMap.create(functionDefinitions);

    return {
        rawResponse,
        schemata,
        categories: categoryMap,
        passthrough,
        functionDefinitions,
        categoriesMissingNodes,
        packages: rawResponse[1],
        fromCache,
    };
};

const processCachedData = (cachedData: BackendData | undefined): NodesInfo | undefined => {
    if (cachedData === undefined) {
        return undefined;
    }
    try {
        return processBackendResponse(cachedData, true);
    } catch (error) {
        log.error('Unable to process the cached node list; waiting for the backend.', error);
        return undefined;
    }
};

const useNodes = (
    backend: Backend,
    backendDownRef: Readonly<MutableRefObject<boolean>>,
    nodesLoadedRef: Readonly<MutableRefObject<boolean>>,
    cachedData: BackendData | undefined
) => {
    const isBackendIntentionallyDown = backendDownRef.current;

    const { t } = useTranslation();
    const { sendAlert, forgetAlert } = useContext(AlertBoxContext);

    // Until the backend's list arrives, the editor works with the list of the last start.
    const [nodesInfo, setNodesInfo] = useState(() => processCachedData(cachedData));

    const queryClient = useQueryClient();
    const refreshNodes = useCallback(() => {
        queryClient.invalidateQueries({ queryKey: ['nodes'] }).catch(log.error);
    }, [queryClient]);

    const [periodicRefresh, setPeriodicRefresh] = useState(false);
    useAsyncEffect(
        () => ({
            supplier: () => ipcRenderer.invoke('refresh-nodes'),
            successEffect: setPeriodicRefresh,
        }),
        []
    );

    const nodesQuery = useQuery({
        queryKey: ['nodes', backend.url],
        queryFn: async (): Promise<BackendData> => {
            // Spin until the backend has loaded all of its nodes, and while it restarts. The
            // replies wait for every node, and while the nodes load, the window's 6
            // connections to the backend serve the previews of the nodes that have loaded.
            while (!nodesLoadedRef.current || backendDownRef.current) {
                // eslint-disable-next-line no-await-in-loop
                await delay(100);
            }

            return Promise.all([
                backend.nodes().catch((e) => {
                    log.error(`Failed to fetch nodes from backend: ${String(e)}`);
                    throw e;
                }),
                backend.packages().catch((e) => {
                    log.error(`Failed to fetch packages from backend: ${String(e)}`);
                    throw e;
                }),
            ]);
        },
        cacheTime: 0,
        retry: 25,
        refetchOnWindowFocus: false,
        refetchInterval: periodicRefresh ? 1000 * 3 : false,
    });

    let nodeQueryError: unknown;
    if (nodesQuery.status === 'error') {
        nodeQueryError = nodesQuery.error;
    } else if (nodesQuery.failureCount > 1) {
        nodeQueryError = 'Failed to fetch backend nodes.';
    }

    const lastErrorAlert = useRef<AlertId>();
    const forgetLastErrorAlert = useCallback(() => {
        if (lastErrorAlert.current !== undefined) {
            forgetAlert(lastErrorAlert.current);
            lastErrorAlert.current = undefined;
        }
    }, [forgetAlert]);

    useEffect(() => {
        if (nodesQuery.status === 'success') {
            forgetLastErrorAlert();
            const rawResponse = nodesQuery.data;
            ipcRenderer.send('update-ui-cache', rawResponse);
            setNodesInfo((prev) => {
                if (prev && isDeepEqual(prev.rawResponse, rawResponse)) {
                    return prev.fromCache ? { ...prev, fromCache: false } : prev;
                }

                try {
                    return processBackendResponse(rawResponse);
                } catch (e) {
                    log.error(e);
                    forgetLastErrorAlert();
                    lastErrorAlert.current = sendAlert({
                        type: AlertType.CRIT_ERROR,
                        title: t(
                            'error.title.unableToProcessNodes',
                            'Unable to process backend nodes.'
                        ),
                        message: `${t(
                            'error.message.criticalBackend',
                            'A critical error occurred while processing the node data returned by the backend.'
                        )}\n\n${String(e)}`,
                    });
                }

                return prev;
            });
        }
    }, [nodesQuery.status, nodesQuery.data, sendAlert, forgetLastErrorAlert, t]);

    useEffect(() => {
        if (nodeQueryError && !isBackendIntentionallyDown) {
            const message =
                nodeQueryError instanceof Error ? nodeQueryError.message : String(nodeQueryError);
            forgetLastErrorAlert();
            lastErrorAlert.current = sendAlert({
                type: AlertType.CRIT_ERROR,
                message: `${t(
                    'error.message.criticalBackend',
                    'A critical error occurred while processing the node data returned by the backend.'
                )}\n\n${t('error.error', 'Error')}: ${message}`,
            });
        }
    }, [nodeQueryError, sendAlert, forgetLastErrorAlert, t, isBackendIntentionallyDown]);

    const scope = useMemo(() => {
        // function definitions all use the same scope, so just pick any one of them
        return [...(nodesInfo?.functionDefinitions.values() ?? [])][0]?.scope ?? getChainnerScope();
    }, [nodesInfo?.functionDefinitions]);
    const schemaInputs = useMemo(
        () => new SchemaInputsMap(nodesInfo?.schemata.schemata ?? []),
        [nodesInfo?.schemata]
    );

    const nodesFromCache = nodesInfo?.fromCache ?? false;
    let connectionState: 'connecting' | 'connected' | 'failed' = 'connecting';
    if (nodesQuery.status === 'error') {
        connectionState = 'failed';
    } else if (nodesInfo !== undefined) {
        connectionState = 'connected';
    }

    return {
        nodesInfo,
        schemaInputs,
        scope,
        refreshNodes,
        connectionState,
        nodesFromCache,
    };
};

const useFeatureStates = (backend: Backend, areNodesLoaded: boolean) => {
    const [featureStates, setFeatureStates] = useState<readonly FeatureState[]>(EMPTY_ARRAY);

    // The reply waits for every node, as the node list does.
    const featuresQuery = useQuery({
        queryKey: ['features', backend.url],
        enabled: areNodesLoaded,
        queryFn: async () => {
            try {
                return await backend.features();
            } catch (error) {
                log.error(`Failed to fetch features from backend: ${String(error)}`);
                throw error;
            }
        },
        retry: true,
        refetchOnWindowFocus: true,
        refetchInterval: 60 * 1000, // refetch every minute
    });

    const { refetch } = featuresQuery;
    const refreshFeatureStates = useCallback(async (): Promise<void> => {
        await refetch().catch(log.error);
    }, [refetch]);

    useEffect(() => {
        if (featuresQuery.status === 'success') {
            const rawResponse = featuresQuery.data;
            setFeatureStates((prev) => {
                return isDeepEqual(prev, rawResponse) ? prev : rawResponse;
            });
        }
    }, [featuresQuery.status, featuresQuery.data]);

    return {
        featureStates,
        refreshFeatureStates,
    };
};

/**
 * The setup events of the backend at `url`, over one connection for the app's life: whether
 * the backend accepts requests (backend-ready), which nodes it has loaded while it loads them
 * (backend-nodes-registered), and whether it has loaded all of them (backend-nodes-loaded);
 * the event source also carries the progress of dependency installs.
 * Until the backend listens, and after it restarts, the connection is tried again every 100 ms,
 * not after EventSource's own delay of a few seconds.
 */
const useBackendSetupEvents = (url: string) => {
    const [isBackendReady, setIsBackendReady] = useState(false);
    const [registeredSchemata, setRegisteredSchemata] = useState<ReadonlySet<SchemaId>>(EMPTY_SET);
    const [areNodesLoaded, setAreNodesLoaded] = useState(false);
    const [setupEventSource, setSetupEventSource] = useState<BackendEventSource | null>(null);
    useEffect(() => {
        let source: EventSource | undefined;
        let retry: ReturnType<typeof setTimeout> | undefined;
        const connect = () => {
            const eventSource = new EventSource(`${url}/setup-sse`, { withCredentials: true });
            source = eventSource;
            setSetupEventSource(eventSource);
            eventSource.addEventListener('backend-ready', () => setIsBackendReady(true));
            eventSource.addEventListener('backend-nodes-registered', (e: MessageEvent<string>) => {
                const { schemaIds } = JSON.parse(
                    e.data
                ) as BackendEventMap['backend-nodes-registered'];
                setRegisteredSchemata(new Set(schemaIds));
            });
            eventSource.addEventListener('backend-nodes-loaded', () => setAreNodesLoaded(true));
            eventSource.onerror = () => {
                eventSource.close();
                retry = setTimeout(connect, 100);
            };
        };
        connect();
        return () => {
            clearTimeout(retry);
            source?.close();
        };
    }, [url]);
    const isNodeLoaded = useCallback(
        (schemaId: SchemaId) => areNodesLoaded || registeredSchemata.has(schemaId),
        [areNodesLoaded, registeredSchemata]
    );
    return { isBackendReady, areNodesLoaded, isNodeLoaded, setupEventSource };
};

export const BackendProvider = memo(
    ({ url, pythonInfo, cachedData, children }: React.PropsWithChildren<BackendProviderProps>) => {
        const backend = getBackend(url);

        const [ownsBackend, setOwnsBackend] = useState<boolean>(false);
        const ownsBackendRef = useRef(ownsBackend);
        useAsyncEffect(
            () => ({
                supplier: () => ipcRenderer.invoke('owns-backend'),
                successEffect: (value) => {
                    setOwnsBackend(value);
                    ownsBackendRef.current = value;
                },
            }),
            []
        );

        const backendDownRef = useRef(false);
        const restartPromiseRef = useRef<Promise<void>>();
        const needsNewRestartRef = useRef(false);
        const nodesLoadedRef = useRef(false);

        const { isBackendReady, areNodesLoaded, isNodeLoaded, setupEventSource } =
            useBackendSetupEvents(url);
        useEffect(() => {
            nodesLoadedRef.current = areNodesLoaded;
        }, [areNodesLoaded]);

        const { nodesInfo, schemaInputs, scope, refreshNodes, connectionState, nodesFromCache } =
            useNodes(backend, backendDownRef, nodesLoadedRef, cachedData);
        const { featureStates, refreshFeatureStates } = useFeatureStates(backend, areNodesLoaded);

        const featureStatesMaps = useMemo((): ReadonlyMap<FeatureId, FeatureState> => {
            return new Map(
                featureStates.map((featureState) => [featureState.featureId, featureState])
            );
        }, [featureStates]);
        const featuresMaps = useMemo((): ReadonlyMap<FeatureId, Feature> => {
            if (nodesInfo === undefined) return EMPTY_MAP;
            return new Map(
                nodesInfo.packages
                    .flatMap((p) => p.features)
                    .map((feature) => [feature.id, feature])
            );
        }, [nodesInfo]);

        const restart = useCallback((): Promise<void> => {
            if (!ownsBackendRef.current) {
                // we don't own the backend, so we can't restart it
                return Promise.resolve();
            }

            if (restartPromiseRef.current) {
                // another promise is currently restarting the backend, so we just request another restart
                needsNewRestartRef.current = true;
                return restartPromiseRef.current;
            }

            backendDownRef.current = true;
            restartPromiseRef.current = (async () => {
                let error;
                try {
                    do {
                        needsNewRestartRef.current = false;
                        try {
                            backend.abort();
                            // eslint-disable-next-line no-await-in-loop
                            await ipcRenderer.invoke('restart-backend');
                            error = null;
                        } catch (e) {
                            error = e;
                        }
                        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
                    } while (needsNewRestartRef.current);
                } finally {
                    // Done. At this point, the backend either restarted or failed trying
                    backendDownRef.current = false;
                    restartPromiseRef.current = undefined;
                }

                refreshNodes();
                refreshFeatureStates().catch(log.error);

                if (error !== null) {
                    throw error instanceof Error ? error : new Error(String(error));
                }
            })();
            return restartPromiseRef.current;
        }, [backend, refreshNodes, refreshFeatureStates]);

        const value = useMemoObject<BackendContextState>({
            url,
            backend,
            ownsBackend,
            schemata: nodesInfo?.schemata ?? SchemaMap.EMPTY,
            schemaInputs,
            pythonInfo,
            categories: nodesInfo?.categories ?? CategoryMap.EMPTY,
            categoriesMissingNodes: nodesInfo?.categoriesMissingNodes ?? EMPTY_ARRAY,
            packages: nodesInfo?.packages ?? EMPTY_ARRAY,
            features: featuresMaps,
            passthrough: nodesInfo?.passthrough ?? PassthroughMap.EMPTY,
            functionDefinitions: nodesInfo?.functionDefinitions ?? EMPTY_MAP,
            scope,
            featureStates: featureStatesMaps,
            refreshFeatureStates,
            backendDownRef,
            restart,
            connectionState,
            isBackendReady,
            nodesFromCache,
            isNodeLoaded,
            setupEventSource,
            refreshNodes,
        });

        return <BackendContext.Provider value={value}>{children}</BackendContext.Provider>;
    }
);
