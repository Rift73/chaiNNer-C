import { useCallback, useEffect, useMemo, useRef } from 'react';

export interface EventBacklog<T> {
    readonly push: (event: T) => void;
    readonly processAll: () => void;
    /** Drops the events waiting that `isDropped` picks. */
    readonly discard: (isDropped: (event: T) => boolean) => void;
}

export interface BacklogOption<T> {
    process: (event: T[]) => void;
    interval: number;
}

export const useEventBacklog = <T>({ process, interval }: BacklogOption<T>): EventBacklog<T> => {
    const backlogRef = useRef<T[]>([]);
    const push = useCallback((event: T): void => {
        backlogRef.current.push(event);
    }, []);

    const processRef = useRef(process);
    useEffect(() => {
        processRef.current = process;
    }, [process]);

    const processAll = useCallback(() => {
        if (backlogRef.current.length > 0) {
            const backlog = backlogRef.current;
            backlogRef.current = [];
            processRef.current(backlog);
        }
    }, []);

    const discard = useCallback((isDropped: (event: T) => boolean) => {
        backlogRef.current = backlogRef.current.filter((event) => !isDropped(event));
    }, []);

    useEffect(() => {
        const timeout = setInterval(processAll, interval);
        return () => clearInterval(timeout);
    }, [processAll, interval]);

    return useMemo(() => ({ push, processAll, discard }), [push, processAll, discard]);
};
