import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useReactFlow } from 'reactflow';
import { useContext, useContextSelector } from 'use-context-selector';
import { EdgeData, NodeData, OutputId, SchemaId } from '../../../common/common-types';
import { log } from '../../../common/log';
import { BackendContext } from '../../contexts/BackendContext';
import { ExecutionContext } from '../../contexts/ExecutionContext';
import { GlobalContext } from '../../contexts/GlobalNodeState';
import { getSeekRender } from '../../helpers/imageCompare';
import {
    UpstreamSeek,
    clampFrame,
    getUpstreamSeekVideo,
    runThisFrameOverrides,
    seekAllTo,
} from '../../helpers/videoSeek';
import { useRunThisFrame } from '../../hooks/useRunThisFrame';
import { useSeekPreview } from '../../hooks/useSeekPreview';
import { InputContainer } from '../inputs/InputContainer';
import { SeekBar } from '../inputs/VideoSeekInput';

interface ViewSeekBarProps {
    readonly nodeId: string;
    /** The seek bars before the node, at least one. */
    readonly seeks: readonly UpstreamSeek[];
    readonly isLocked: boolean;
    /** A loop source before the node runs all its items: Run this frame cannot run. */
    readonly hasFullLoop: boolean;
    /** The image on side A of the node's viewer (getSideA), which a move of the bar renders. */
    readonly sideA: OutputId | undefined;
}

/**
 * A seek bar of a node's own that moves the seek bars before it (View Image's): it shows their
 * frame (the first one's when they differ) within the shortest of their videos, and sets every
 * one to its frame, each clamped to its own video, when a gesture ends (a drag is released, steps
 * stop, a typed frame) and when Run this frame runs. It has no value of its own; Run this frame
 * runs the frame it shows. A move renders the frame of the viewer's side A image alone
 * (getSeekRender; with the image the node requires) and leaves the bars before it where they are,
 * whose nodes would each render their own Preview, so a drag or steps in a row run only what
 * feeds side A; when the gesture ends, one more render of the frame renders every image
 * connected, as Run this frame does, and the viewer lists them all again.
 */
export const ViewSeekBar = memo(
    ({ nodeId, seeks, isLocked, hasFullLoop, sideA }: ViewSeekBarProps) => {
        const { t } = useTranslation();
        const { setNodeInputValue } = useContext(GlobalContext);
        const { getNode, getNodes, getEdges } = useReactFlow<NodeData, EdgeData>();
        const schemata = useContextSelector(BackendContext, (c) => c.schemata);
        // every node's last reported outputs, for the other seek bars' videos, read per frame
        const getOutputData = useContextSelector(GlobalContext, (c) => c.getOutputData);

        const video = useMemo(() => getUpstreamSeekVideo(seeks), [seeks]);
        const [first] = seeks;
        const committed = video ? clampFrame(first.frame, video.frameCount) : first.frame;

        // the frame on every bar it moves, every other seek bar's node on its own frame, as Run
        // this frame runs them, and of the viewer's images side A's alone, or all at a gesture's end
        const overridesFor = useCallback(
            (n: number, endsGesture: boolean) => {
                if (sideA === undefined) return undefined;
                const nodes = getNodes();
                const edges = getEdges();
                const getSchema = (schemaId: SchemaId) => schemata.get(schemaId);
                const viewer = getSeekRender(
                    nodeId,
                    endsGesture ? 'all' : sideA,
                    nodes,
                    edges,
                    getSchema
                );
                if (viewer === undefined) return undefined;
                const overrides = runThisFrameOverrides(
                    nodes.map((node) => node.data),
                    edges,
                    getSchema,
                    (id) => getOutputData(id)?.types,
                    new Map(seekAllTo(seeks, n).map((seek) => [seek.nodeId, seek.frame]))
                );
                return overrides.set(nodeId, viewer);
            },
            [sideA, getNodes, getEdges, schemata, nodeId, getOutputData, seeks]
        );
        const moveTo = useCallback(
            (n: number) => {
                for (const seek of seekAllTo(seeks, n)) {
                    // the bar can go away with the nodes it moves (their chain closed)
                    if (getNode(seek.nodeId) !== undefined) {
                        setNodeInputValue(seek.nodeId, seek.inputId, seek.frame);
                    }
                }
            },
            [seeks, getNode, setNodeInputValue]
        );
        const kind = useMemo(() => getNode(nodeId)?.data.schemaId, [getNode, nodeId]);
        const previewOptions = useMemo(() => ({ kind }), [kind]);
        const showPreview = useSeekPreview(nodeId, overridesFor, previewOptions);
        const renderPreview = useContextSelector(ExecutionContext, (c) => c.renderPreview);
        const endSeek = useCallback(
            (n: number, isLeaving = false) => {
                moveTo(n);
                if (!isLeaving) {
                    showPreview(n, false, true);
                    return;
                }
                // The bar goes away during a gesture: the viewer's picture would keep its
                // previews alone. Every image renders complete now (no queue renders any more).
                const all = getNode(nodeId) === undefined ? undefined : overridesFor(n, true);
                if (all !== undefined) renderPreview(nodeId, all).catch(log.error);
            },
            [moveTo, showPreview, getNode, nodeId, overridesFor, renderPreview]
        );
        // The press runs the frame the bar shows: it first moves every bar there, as the end of a
        // gesture does, then runs what the viewer shows (the node and what it depends on, no Save node).
        const run = useRunThisFrame();
        const runThisFrame = useCallback(
            (n: number) => {
                moveTo(n);
                const moved = seekAllTo(seeks, n);
                run(new Map(moved.map((seek) => [seek.nodeId, seek.frame])), {
                    nodeId,
                    kind: 'up-to',
                });
            },
            [moveTo, seeks, run, nodeId]
        );

        return (
            <InputContainer passthroughIgnored={false}>
                <SeekBar
                    cannotRunText={
                        hasFullLoop
                            ? t(
                                  'inputs.videoSeek.loopRunsAllItems',
                                  'Run this frame is off here: a loop source without a seek bar (such as Load Images) also feeds this viewer.'
                              )
                            : undefined
                    }
                    committed={committed}
                    def={first.input.def}
                    endSeek={endSeek}
                    isDisabled={isLocked || video === undefined}
                    label={first.input.label}
                    runThisFrame={runThisFrame}
                    setSeek={showPreview}
                    video={video}
                    waitingText={t(
                        'inputs.videoSeek.waitingForVideos',
                        'The seek bar works once the nodes before this one have read their videos.'
                    )}
                />
            </InputContainer>
        );
    }
);
