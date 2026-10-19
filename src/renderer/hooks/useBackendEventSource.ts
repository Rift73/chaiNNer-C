import {
    EventSourceEvent,
    EventSourceStatus,
    useEventSource,
    useEventSourceListener,
} from '@react-nano/use-event-source';
import { BackendEventMap } from '../../common/Backend';
import { log } from '../../common/log';

export type BackendEventSource = EventSource & { readonly __backend?: never };

// Both connect once the backend accepts requests: EventSource would try a failed connection again
// only after a few seconds. The execution events of runs started earlier wait in the backend.

/** The backend's execution events. */
export const useBackendEventSource = (
    url: string,
    isBackendReady: boolean
): readonly [BackendEventSource | null, EventSourceStatus] => {
    return useEventSource(isBackendReady ? `${url}/sse` : '', true);
};

/** The backend's setup events, e.g. the progress of a dependency install. */
export const useBackendSetupEventSource = (
    url: string,
    isBackendReady: boolean
): readonly [BackendEventSource | null, EventSourceStatus] => {
    return useEventSource(isBackendReady ? `${url}/setup-sse` : '', true);
};

export type BackendEventSourceListener<T extends keyof BackendEventMap> = (
    e: BackendEventMap[T] | undefined,
    event: EventSourceEvent
) => void;

export const useBackendEventSourceListener = <T extends keyof BackendEventMap>(
    source: BackendEventSource | null,
    type: T,
    listener: BackendEventSourceListener<T>
): void => {
    useEventSourceListener(
        source,
        [type],
        (event) => {
            let parsed: BackendEventMap[T] | undefined;
            try {
                parsed = JSON.parse(event.data) as BackendEventMap[T];
            } catch (error) {
                log.error(`The response data for event ${type} was invalid JSON: `, error);
            }

            try {
                listener(parsed, event);
            } catch (error) {
                log.error(
                    `Listener for event ${type} and data ${
                        parsed === undefined ? 'undefined' : JSON.stringify(parsed)
                    } errored: `,
                    error
                );
            }
        },
        [source, type, listener]
    );
};
