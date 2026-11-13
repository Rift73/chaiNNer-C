import { useCallback, useEffect, useRef, useState } from 'react';
import { useContext, useContextSelector } from 'use-context-selector';
import { InputData } from '../../common/common-types';
import { log } from '../../common/log';
import { AlertBoxContext, AlertType } from '../contexts/AlertBoxContext';
import { BackendContext } from '../contexts/BackendContext';
import { ExecutionContext, ExecutionStatus } from '../contexts/ExecutionContext';

/**
 * Renders the Preview of a node whose seek bar moves, for a node that a change of its inputs does
 * not run by itself (one with connections, such as Clip Frames behind its clip). The returned
 * function asks for a frame: it is rendered through the nodes the node depends on, with the
 * node's inputs overridden by `overridesFor(frame)`, as Run this frame renders it, but nothing
 * after the node runs (ExecutionContext's renderPreview: no run of the chain).
 *
 * One render at a time: a frame asked for during a render waits for it and replaces the frame
 * waiting before it, so only the latest is rendered. While a run of the chain goes on or waits,
 * nothing is asked for or rendered. A failed render is alerted once, and nothing more renders
 * until a new gesture (`startsGesture`: a drag begins, a step, a typed frame) asks for a frame.
 */
export const useSeekPreview = (
    nodeId: string | undefined,
    overridesFor: (frame: number) => InputData
): ((frame: number, startsGesture: boolean) => void) => {
    const { sendAlert } = useContext(AlertBoxContext);
    const isBackendReady = useContextSelector(BackendContext, (c) => c.isBackendReady);
    const status = useContextSelector(ExecutionContext, (c) => c.status);
    const renderPreview = useContextSelector(ExecutionContext, (c) => c.renderPreview);
    const canRun = isBackendReady && status === ExecutionStatus.READY;

    const [isRendering, setIsRendering] = useState(false);
    const [asked, setAsked] = useState(0);
    const wantedRef = useRef<number>();
    const failedRef = useRef(false);

    useEffect(() => {
        const frame = wantedRef.current;
        if (nodeId === undefined || frame === undefined || isRendering || !canRun) return;
        wantedRef.current = undefined;
        setIsRendering(true);
        renderPreview(nodeId, overridesFor(frame))
            .then((render) => {
                if (render.kind === 'failed') {
                    failedRef.current = true;
                    wantedRef.current = undefined;
                    sendAlert({ type: AlertType.ERROR, message: render.message });
                }
            })
            .finally(() => setIsRendering(false))
            .catch(log.error);
    }, [asked, isRendering, canRun, nodeId, renderPreview, overridesFor, sendAlert]);

    return useCallback(
        (frame: number, startsGesture: boolean) => {
            if (startsGesture) failedRef.current = false;
            if (failedRef.current || !canRun) return;
            wantedRef.current = frame;
            setAsked((n) => n + 1);
        },
        [canRun]
    );
};
