import { useCallback, useEffect, useRef, useState } from 'react';
import { useContext, useContextSelector } from 'use-context-selector';
import { InputData } from '../../common/common-types';
import { log } from '../../common/log';
import { AlertBoxContext, AlertType } from '../contexts/AlertBoxContext';
import { BackendContext } from '../contexts/BackendContext';
import { ExecutionContext, ExecutionStatus, PreviewRender } from '../contexts/ExecutionContext';

// endsGesture: the render adds what a gesture's end adds (overridesFor's)
type SeekRender = (
    nodeId: string,
    overrides: ReadonlyMap<string, InputData>,
    endsGesture: boolean
) => Promise<PreviewRender>;

interface SeekPreviewOptions {
    /** How a frame renders; by default through the nodes the node depends on (renderPreview). */
    readonly render?: SeekRender;
    /** What the gesture's log line names the node by. */
    readonly kind?: string;
    /**
     * Whether a gesture's end on a frame not rendered yet renders the end alone, not the frame as
     * a move first (a node by itself: the end is no heavier than a move).
     */
    readonly endWithoutMove?: () => boolean;
}

/** A gesture's renders, for its log line. */
interface GestureStats {
    readonly start: number;
    asked: number;
    shown: number;
    /** Per frame shown, the time from its request to it (ms). */
    readonly times: number[];
    /** Per frame shown, the node's own time in the backend (ms), when the render told it. */
    readonly backendTimes: number[];
    /** The preview widths the renders asked for, as `a+b`. */
    readonly sizes: Set<string>;
    ended: boolean;
}

const median = (values: readonly number[]): number => {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
const ms = (value: number) => `${Math.round(value)} ms`;

const logGesture = (kind: string, nodeId: string, stats: GestureStats) => {
    const { times, backendTimes } = stats;
    const timing =
        times.length > 0
            ? `, request to frame median ${ms(median(times))}, worst ${ms(Math.max(...times))}`
            : '';
    const backend = backendTimes.length > 0 ? `, backend median ${ms(median(backendTimes))}` : '';
    const sizes = stats.sizes.size > 0 ? `, preview sizes ${[...stats.sizes].join(', ')}` : '';
    log.info(
        `Seek of ${kind} ${nodeId}: ${stats.asked} frames asked, ${stats.shown} shown in ${ms(
            performance.now() - stats.start
        )}${timing}${backend}${sizes}`
    );
};

/**
 * Renders the Preview of a node whose seek bar moves: the returned function asks for a frame. It
 * is rendered with the inputs of the nodes overridden by `overridesFor(frame, endsGesture)` (by
 * node id): by default through the nodes the node depends on, as Run this frame renders it, but
 * nothing after the node runs (ExecutionContext's renderPreview: no run of the chain), or by
 * `options.render` (a node without connections by itself: renderAlone). `endsGesture`: the
 * gesture that moves the bar ends at the frame (a drag is released, steps stop, a typed frame), so
 * the frame may render more than the frames of the gesture did; when the last render started was
 * of another frame, the end's frame first renders as a move does, then with what the end adds. A
 * frame for which `overridesFor` gives nothing is not rendered.
 *
 * One render at a time: a frame asked for during a render waits for it and replaces the frame
 * waiting before it (a gesture's end too), so only the latest is rendered; a render another seek
 * bar asked for is waited for, so the last frame asked for is rendered once it ends. A render in
 * flight is not abandoned: a frame asked for meanwhile renders after it. While a run of the chain
 * goes on or waits, nothing is asked for or rendered: the run renders the frame. A failed render
 * is alerted once, and nothing more renders, a gesture's end neither, until a new gesture
 * (`startsGesture`: a drag begins, a step, a typed frame) asks for a frame. Once a gesture has
 * ended and its renders are done, one line in the log tells its frames and their times. The
 * function returns whether a render of the frame is to start (a gesture's end: whether its frame
 * still renders).
 */
export const useSeekPreview = (
    nodeId: string | undefined,
    overridesFor: (
        frame: number,
        endsGesture: boolean
    ) => ReadonlyMap<string, InputData> | undefined,
    options: SeekPreviewOptions = {}
): ((frame: number, startsGesture: boolean, endsGesture?: boolean) => boolean) => {
    const { sendAlert } = useContext(AlertBoxContext);
    const isBackendReady = useContextSelector(BackendContext, (c) => c.isBackendReady);
    const status = useContextSelector(ExecutionContext, (c) => c.status);
    const renderPreview = useContextSelector(ExecutionContext, (c) => c.renderPreview);
    const { render, endWithoutMove } = options;
    const kind = options.kind ?? 'node';
    const canRun = isBackendReady && status === ExecutionStatus.READY;

    const [isRendering, setIsRendering] = useState(false);
    const [asked, setAsked] = useState(0);
    const wantedRef = useRef<{ frame: number; endsGesture: boolean }>();
    // the frame of the last render started
    const lastStartedRef = useRef<number>();
    const failedRef = useRef(false);
    const statsRef = useRef<GestureStats>();

    useEffect(() => {
        if (nodeId === undefined || isRendering) return;
        const wanted = wantedRef.current;
        const stats = statsRef.current;
        if (wanted === undefined || !canRun) {
            if (stats?.ended) {
                statsRef.current = undefined;
                logGesture(kind, nodeId, stats);
            }
            return;
        }
        // A gesture's end on a frame not rendered yet renders it as a move first, which shows
        // as soon as a move's would, and waits (unless a newer frame replaces it) to render
        // again with what the end adds; unless the bar has it render the end alone.
        const first =
            wanted.endsGesture &&
            lastStartedRef.current !== wanted.frame &&
            endWithoutMove?.() !== true
                ? overridesFor(wanted.frame, false)
                : undefined;
        if (first === undefined) wantedRef.current = undefined;
        const overrides = first ?? overridesFor(wanted.frame, wanted.endsGesture);
        const endRender = first === undefined && wanted.endsGesture;
        if (overrides === undefined) {
            // nothing more to render: the effect runs again for the log line
            setAsked((n) => n + 1);
            return;
        }
        lastStartedRef.current = wanted.frame;
        setIsRendering(true);
        const requested = performance.now();
        (render
            ? render(nodeId, overrides, endRender)
            : renderPreview(nodeId, overrides, !endRender)
        )
            .then((result) => {
                if (result.kind === 'failed') {
                    failedRef.current = true;
                    wantedRef.current = undefined;
                    sendAlert({ type: AlertType.ERROR, message: result.message });
                } else if (result.kind === 'rendered' && stats !== undefined) {
                    stats.shown += 1;
                    stats.times.push(performance.now() - requested);
                    if (result.backendTime !== undefined) {
                        stats.backendTimes.push(result.backendTime * 1000);
                    }
                    if (result.sizes !== undefined) stats.sizes.add(result.sizes.join('+'));
                }
            })
            .finally(() => setIsRendering(false))
            .catch(log.error);
    }, [
        asked,
        isRendering,
        canRun,
        nodeId,
        render,
        renderPreview,
        overridesFor,
        sendAlert,
        kind,
        endWithoutMove,
    ]);

    return useCallback(
        (frame: number, startsGesture: boolean, endsGesture = false) => {
            if (startsGesture) failedRef.current = false;
            if (failedRef.current || !canRun) return false;
            if (startsGesture || statsRef.current === undefined) {
                statsRef.current = {
                    start: performance.now(),
                    asked: 0,
                    shown: 0,
                    times: [],
                    backendTimes: [],
                    sizes: new Set(),
                    ended: false,
                };
            }
            if (endsGesture) {
                statsRef.current.ended = true;
            } else {
                statsRef.current.asked += 1;
            }
            wantedRef.current = { frame, endsGesture };
            setAsked((n) => n + 1);
            return (
                !endsGesture ||
                lastStartedRef.current !== frame ||
                overridesFor(frame, true) !== undefined
            );
        },
        [canRun, overridesFor]
    );
};
