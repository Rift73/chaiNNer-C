import {
    Box,
    Button,
    Flex,
    HStack,
    IconButton,
    MenuItem,
    MenuList,
    Spacer,
    Text,
    Tooltip,
    VStack,
} from '@chakra-ui/react';
import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
    MdContentCopy,
    MdContentPaste,
    MdFastForward,
    MdFastRewind,
    MdFirstPage,
    MdLastPage,
    MdNavigateBefore,
    MdNavigateNext,
    MdPlayArrow,
} from 'react-icons/md';
import { useReactFlow } from 'reactflow';
import { createContext, useContext, useContextSelector } from 'use-context-selector';
import { EdgeData, InputData, NodeData, OutputTypes } from '../../../common/common-types';
import { log } from '../../../common/log';
import { BackendContext } from '../../contexts/BackendContext';
import { ExecutionContext, ExecutionStatus } from '../../contexts/ExecutionContext';
import { FakeNodeContext } from '../../contexts/FakeExampleContext';
import { GlobalContext, GlobalVolatileContext } from '../../contexts/GlobalNodeState';
import { InputContext } from '../../contexts/InputContext';
import { parsePastedNumber } from '../../helpers/pastedNumber';
import { LINEAR_SCALE } from '../../helpers/sliderScale';
import {
    FRAMES_RANGE,
    SeekStep,
    SeekVideo,
    clampFrame,
    formatTimecode,
    getNodeSeekVideo,
    oneFrameOverride,
    scrubFrame,
    seekStepForKey,
    stepFrame,
} from '../../helpers/videoSeek';
import { useContextMenu } from '../../hooks/useContextMenu';
import { useInputRefactor } from '../../hooks/useInputRefactor';
import { useRunThisFrame } from '../../hooks/useRunThisFrame';
import { useSeekPreview } from '../../hooks/useSeekPreview';
import { AdvancedNumberInput } from './elements/AdvanceNumberInput';
import { SliderStyle, StyledSlider } from './elements/StyledSlider';
import { WithoutLabel } from './InputContainer';
import { InputProps } from './props';

/** Steps by the buttons or the wheel this close (ms) are one gesture; it ends after the last. */
const STEP_END_DELAY = 400;

export interface SeekBarLayout {
    /**
     * Set by a large image output's maximized view: the bar spans the view in one row and its keys
     * work anywhere in the view.
     */
    readonly isWide: boolean;
    /** The node's Frames, Start and End inputs, drawn with the bar when it is under the Preview. */
    readonly companions?: React.ReactNode;
}
export const SeekBarLayoutContext = createContext<SeekBarLayout>({ isWide: false });

interface SeekButtonProps {
    label: string;
    icon: React.ReactElement;
    isDisabled: boolean;
    onClick: () => void;
}

const SeekButton = memo(({ label, icon, isDisabled, onClick }: SeekButtonProps) => (
    <Tooltip
        closeOnClick
        hasArrow
        borderRadius={8}
        label={label}
        openDelay={500}
    >
        <IconButton
            aria-label={label}
            className="nodrag"
            h="1.5rem"
            icon={icon}
            isDisabled={isDisabled}
            minWidth={0}
            size="xs"
            variant="ghost"
            w="1.5rem"
            onClick={onClick}
        />
    </Tooltip>
));

export interface SeekBarProps {
    /** The frame the bar's value holds, and the frame of its reset. */
    readonly committed: number;
    readonly def: number;
    readonly label: string;
    readonly video: SeekVideo | undefined;
    readonly isDisabled: boolean;
    /**
     * Asks for frame `n` during a gesture, at each move of a drag, each step and a typed frame:
     * the frame is shown, the value is set at the gesture's end (endSeek). startsGesture: a drag
     * begins, a step, a typed frame (a drag's last frame is no new one).
     */
    readonly setSeek: (n: number, startsGesture: boolean) => void;
    /**
     * The gesture that set frame `n` (with setSeek) has ended: a drag is released, a typed or
     * pasted frame at once; steps in a row are one gesture, which ends when the key that steps is
     * let go (or the window loses the focus), or STEP_END_DELAY ms after the last step of the
     * buttons or the wheel; also when the bar goes away during steps (`isLeaving`: nothing of it
     * renders any more).
     */
    readonly endSeek?: (n: number, isLeaving?: boolean) => void;
    /** Run this frame, at frame `n`. */
    readonly runThisFrame: (n: number) => void;
    /**
     * Set start and Set end: sets the start (`isIn`) or the end of the frame range at frame `n`.
     * The bar has the two buttons when it is given.
     */
    readonly setRange?: (isIn: boolean, n: number) => void;
    /** The last items of the bar's context menu. */
    readonly menuItems?: React.ReactNode;
    /** Shown until the video is known. */
    readonly waitingText: string;
    /** Why Run this frame cannot run here: the button is disabled and this shown instead. */
    readonly cannotRunText?: string;
}

/**
 * A seek bar: the slider, the step buttons, the frame box with its timecode, Set start and Set end
 * (with `setRange`) and Run this frame; its keys, and in the maximized view the wheel and the
 * Alt-drag scrub.
 */
export const SeekBar = memo(
    ({
        committed,
        def,
        label,
        video,
        isDisabled,
        setSeek,
        endSeek,
        runThisFrame,
        setRange,
        menuItems,
        waitingText,
        cannotRunText,
    }: SeekBarProps) => {
        const { t } = useTranslation();
        const { isFake } = useContext(FakeNodeContext);
        const { conditionallyInactive } = useContext(InputContext);
        const { isWide, companions } = useContext(SeekBarLayoutContext);
        const isBackendReady = useContextSelector(BackendContext, (c) => c.isBackendReady);
        const status = useContextSelector(ExecutionContext, (c) => c.status);

        // The frame shown: ahead of the value while the slider is dragged.
        const [frame, setFrame] = useState(committed);
        const [frameString, setFrameString] = useState(String(committed));
        const draggingRef = useRef(false);
        useEffect(() => {
            if (!draggingRef.current) setFrame(committed);
        }, [committed]);
        useEffect(() => {
            setFrameString(String(frame));
        }, [frame]);

        const commit = useCallback(
            (n: number, startsGesture = true) => {
                if (video === undefined) return;
                const clamped = clampFrame(n, video.frameCount);
                setFrame(clamped);
                setSeek(clamped, startsGesture);
            },
            [video, setSeek]
        );
        // The last step of the steps whose gesture goes on: its frame, and whether a key set it
        // (the gesture ends when the key is let go) or a button or the wheel (STEP_END_DELAY ms
        // after it).
        const pendingStepRef = useRef<{ frame: number; byKey: boolean }>();
        const stepEndTimerRef = useRef<ReturnType<typeof setTimeout>>();

        const dropSteps = useCallback(() => {
            clearTimeout(stepEndTimerRef.current);
            pendingStepRef.current = undefined;
        }, []);
        const endSteps = useCallback(
            (isLeaving = false) => {
                const pending = pendingStepRef.current;
                dropSteps();
                if (pending !== undefined) endSeek?.(pending.frame, isLeaving);
            },
            [dropSteps, endSeek]
        );
        // The frame a drag going on last asked for.
        const draggedRef = useRef<number>();
        // The bar going away during a drag or steps (the maximized view closes) ends them: the
        // value is their last frame.
        const leaveRef = useRef(() => {});
        leaveRef.current = () => {
            const dragged = draggedRef.current;
            if (!draggingRef.current || dragged === undefined) {
                endSteps(true);
                return;
            }
            draggingRef.current = false;
            draggedRef.current = undefined;
            if (video !== undefined) endSeek?.(clampFrame(dragged, video.frameCount), true);
        };
        useEffect(() => () => leaveRef.current(), []);
        // a frame set by a gesture that ends with it (a drag's release, a typed frame)
        const commitEnd = useCallback(
            (n: number, startsGesture = true) => {
                if (video === undefined) return;
                dropSteps();
                commit(n, startsGesture);
                endSeek?.(clampFrame(n, video.frameCount));
            },
            [video, dropSteps, commit, endSeek]
        );
        const stepTo = (to: number, byKey = false) => {
            if (video === undefined) return;
            const n = clampFrame(to, video.frameCount);
            dropSteps();
            commit(n);
            pendingStepRef.current = { frame: n, byKey };
            if (!byKey) stepEndTimerRef.current = setTimeout(endSteps, STEP_END_DELAY);
        };
        const step = (seekStep: SeekStep, byKey = false) => {
            if (video === undefined) return;
            stepTo(stepFrame(seekStep, frame, video.frameCount, video.fps), byKey);
        };
        useEffect(() => {
            if (endSeek === undefined) return undefined;
            const onKeyUp = (event: KeyboardEvent) => {
                const isStepKey = seekStepForKey(event.key, false, 'ten') !== undefined;
                if (isStepKey && pendingStepRef.current?.byKey) endSteps();
            };
            // the window loses the focus while the key is held: its keyup never comes
            const onBlur = () => {
                if (pendingStepRef.current?.byKey) endSteps();
            };
            window.addEventListener('keyup', onKeyUp);
            window.addEventListener('blur', onBlur);
            return () => {
                window.removeEventListener('keyup', onKeyUp);
                window.removeEventListener('blur', onBlur);
            };
        }, [endSeek, endSteps]);

        // A drag asks for its frame at each move; its release sets it.
        // whether the next frame the drag asks for is its first
        const dragStartsRef = useRef(false);
        const onSliderChange = useCallback(
            (n: number) => {
                if (!draggingRef.current) {
                    dragStartsRef.current = true;
                    // the drag's release ends the steps before it too
                    dropSteps();
                }
                draggingRef.current = true;
                draggedRef.current = n;
                setFrame(n);
                setSeek(n, dragStartsRef.current);
                dragStartsRef.current = false;
            },
            [dropSteps, setSeek]
        );
        const onSliderChangeEnd = useCallback(
            (n: number) => {
                draggingRef.current = false;
                draggedRef.current = undefined;
                commitEnd(n, dragStartsRef.current);
                dragStartsRef.current = false;
            },
            [commitEnd]
        );

        // Left/Right step a frame (a second with Shift), Home/End go to the first/last frame,
        // while the seek bar has the focus (but not its frame box, whose text they edit).
        const onKeyDownCapture = (event: React.KeyboardEvent) => {
            if (isDisabled || event.target instanceof HTMLInputElement) return;
            const seekStep = seekStepForKey(event.key, event.shiftKey, 'second');
            if (seekStep === undefined) return;
            // the slider's own keys and the editor's (moving the node) do not apply
            event.preventDefault();
            event.stopPropagation();
            step(seekStep, true);
        };
        // In the maximized view the keys work wherever its focus is, but in a text box, and
        // Shift steps ten frames.
        const stepRef = useRef(step);
        stepRef.current = step;
        useEffect(() => {
            if (!isWide || isDisabled) return;
            const onKeyDown = (event: KeyboardEvent) => {
                if (event.target instanceof HTMLInputElement) return;
                const seekStep = seekStepForKey(event.key, event.shiftKey, 'ten');
                if (seekStep === undefined) return;
                event.preventDefault();
                stepRef.current(seekStep, true);
            };
            window.addEventListener('keydown', onKeyDown);
            return () => window.removeEventListener('keydown', onKeyDown);
        }, [isWide, isDisabled]);
        // An Alt-drag on the maximized picture scrubs at a fixed rate, whatever the video's
        // length; it sets its frame as a slider drag does.
        const scrubRef = useRef({ onSliderChange, onSliderChangeEnd, frame, video });
        scrubRef.current = { onSliderChange, onSliderChangeEnd, frame, video };
        useEffect(() => {
            if (!isWide || isDisabled) return;
            const onPointerDown = (down: PointerEvent) => {
                const { target } = down;
                if (!down.altKey || down.button !== 0 || !(target instanceof Element)) return;
                if (!target.closest('[data-seek-scrub]')) return;
                const { frame: startFrame, video: startVideo } = scrubRef.current;
                if (startVideo === undefined) return;
                down.preventDefault();
                let last = startFrame;
                const onMove = (move: PointerEvent) => {
                    const n = scrubFrame(
                        startFrame,
                        move.clientX - down.clientX,
                        startVideo.frameCount
                    );
                    if (n !== last) {
                        last = n;
                        scrubRef.current.onSliderChange(n);
                    }
                };
                const onUp = () => {
                    window.removeEventListener('pointermove', onMove);
                    window.removeEventListener('pointerup', onUp);
                    scrubRef.current.onSliderChangeEnd(last);
                };
                window.addEventListener('pointermove', onMove);
                window.addEventListener('pointerup', onUp);
            };
            window.addEventListener('pointerdown', onPointerDown);
            return () => window.removeEventListener('pointerdown', onPointerDown);
        }, [isWide, isDisabled]);
        // the wheel over the maximized view's bar steps a frame
        const onWheel = (event: React.WheelEvent) => {
            if (!isWide || isDisabled || event.deltaY === 0) return;
            step(event.deltaY > 0 ? 'frame-forward' : 'frame-back');
        };

        const setRangeEnd = (isIn: boolean) => {
            if (setRange === undefined || video === undefined) return;
            setRange(isIn, clampFrame(frame, video.frameCount));
        };

        const canRun = isBackendReady && status === ExecutionStatus.READY;
        const pressRun = () => {
            if (video === undefined) return;
            runThisFrame(clampFrame(frame, video.frameCount));
        };

        const menu = useContextMenu(() => (
            <MenuList className="nodrag">
                <MenuItem
                    icon={<MdContentCopy />}
                    onClick={() => {
                        navigator.clipboard.writeText(String(frame)).catch(log.error);
                    }}
                >
                    {t('inputs.number.copyText', 'Copy Number')}
                </MenuItem>
                <MenuItem
                    icon={<MdContentPaste />}
                    isDisabled={isDisabled}
                    onClick={() => {
                        navigator.clipboard
                            .readText()
                            .then((clipboardValue) => {
                                const n = parsePastedNumber(clipboardValue, null, null, 0);
                                if (n !== undefined) {
                                    commitEnd(n);
                                }
                            })
                            .catch(log.error);
                    }}
                >
                    {t('inputs.number.paste', 'Paste')}
                </MenuItem>
                {menuItems}
            </MenuList>
        ));

        const sliderStyle = useMemo(
            (): SliderStyle => ({ type: 'label', label, unused: conditionallyInactive }),
            [label, conditionallyInactive]
        );
        // a slider needs two ends: a video of one frame has nothing to seek
        const lastFrame = video ? video.frameCount - 1 : 0;
        const sliderMax = Math.max(1, lastFrame);
        // a clip without a frame rate has frame numbers only
        const timecode = video ? formatTimecode(frame, video.fps) : '-:--:--.---';
        const frameDigits = String(Math.max(0, lastFrame)).length;

        const slider = (
            <Box
                w="full"
                onWheel={onWheel}
            >
                <StyledSlider
                    def={def}
                    isDisabled={isDisabled || lastFrame < 1}
                    max={sliderMax}
                    min={0}
                    scale={LINEAR_SCALE}
                    step={1}
                    style={sliderStyle}
                    tooltip={timecode === undefined ? String(frame) : `${frame} (${timecode})`}
                    value={Math.min(frame, sliderMax)}
                    onChange={onSliderChange}
                    onChangeEnd={onSliderChangeEnd}
                    onContextMenu={menu.onContextMenu}
                />
            </Box>
        );
        const stepButtons = (
            <>
                <SeekButton
                    icon={<MdFirstPage />}
                    isDisabled={isDisabled}
                    label={t('inputs.videoSeek.firstFrame', 'First frame (Home)')}
                    onClick={() => step('first')}
                />
                {isWide ? (
                    <SeekButton
                        icon={<MdFastRewind />}
                        isDisabled={isDisabled}
                        label={t('inputs.videoSeek.tenBack', 'Back 10 frames (Shift+Left)')}
                        onClick={() => step('ten-back')}
                    />
                ) : (
                    <SeekButton
                        icon={<MdFastRewind />}
                        isDisabled={isDisabled}
                        label={t('inputs.videoSeek.secondBack', 'Back 1 second (Shift+Left)')}
                        onClick={() => step('second-back')}
                    />
                )}
                <SeekButton
                    icon={<MdNavigateBefore />}
                    isDisabled={isDisabled}
                    label={t('inputs.videoSeek.frameBack', 'Previous frame (Left)')}
                    onClick={() => step('frame-back')}
                />
                <SeekButton
                    icon={<MdNavigateNext />}
                    isDisabled={isDisabled}
                    label={t('inputs.videoSeek.frameForward', 'Next frame (Right)')}
                    onClick={() => step('frame-forward')}
                />
                {isWide ? (
                    <SeekButton
                        icon={<MdFastForward />}
                        isDisabled={isDisabled}
                        label={t('inputs.videoSeek.tenForward', 'Forward 10 frames (Shift+Right)')}
                        onClick={() => step('ten-forward')}
                    />
                ) : (
                    <SeekButton
                        icon={<MdFastForward />}
                        isDisabled={isDisabled}
                        label={t(
                            'inputs.videoSeek.secondForward',
                            'Forward 1 second (Shift+Right)'
                        )}
                        onClick={() => step('second-forward')}
                    />
                )}
                <SeekButton
                    icon={<MdLastPage />}
                    isDisabled={isDisabled}
                    label={t('inputs.videoSeek.lastFrame', 'Last frame (End)')}
                    onClick={() => step('last')}
                />
            </>
        );
        const rangeButtons = setRange && (
            <>
                <Tooltip
                    closeOnClick
                    hasArrow
                    borderRadius={8}
                    label={t('inputs.videoSeek.setIn', 'Start the frame range here')}
                    openDelay={500}
                >
                    <Button
                        className="nodrag"
                        h="1.5rem"
                        isDisabled={isDisabled}
                        size="xs"
                        variant="outline"
                        onClick={() => setRangeEnd(true)}
                    >
                        {t('inputs.videoSeek.in', 'Set start')}
                    </Button>
                </Tooltip>
                <Tooltip
                    closeOnClick
                    hasArrow
                    borderRadius={8}
                    label={t('inputs.videoSeek.setOut', 'End the frame range here')}
                    openDelay={500}
                >
                    <Button
                        className="nodrag"
                        h="1.5rem"
                        isDisabled={isDisabled}
                        size="xs"
                        variant="outline"
                        onClick={() => setRangeEnd(false)}
                    >
                        {t('inputs.videoSeek.out', 'Set end')}
                    </Button>
                </Tooltip>
            </>
        );
        const frameBox = (
            <>
                <AdvancedNumberInput
                    hideTrailingZeros
                    small
                    controlsStep={1}
                    defaultValue={def}
                    inputString={frameString}
                    inputWidth={`${1.85 + frameDigits * 0.4}rem`}
                    isDisabled={isDisabled}
                    max={Math.max(0, lastFrame)}
                    min={0}
                    precision={0}
                    setInput={commitEnd}
                    setInputString={setFrameString}
                    onContextMenu={menu.onContextMenu}
                    onStep={stepTo}
                />
                {timecode !== undefined && (
                    <Text
                        fontFamily="mono"
                        fontSize="xs"
                        opacity={video ? undefined : 0.5}
                    >
                        {timecode}
                    </Text>
                )}
            </>
        );
        const runButton = (
            <Button
                className="nodrag"
                h="1.5rem"
                isDisabled={isDisabled || !canRun || cannotRunText !== undefined}
                leftIcon={<MdPlayArrow />}
                size="xs"
                variant="outline"
                w={isWide ? undefined : 'full'}
                onClick={pressRun}
            >
                {t('inputs.videoSeek.runThisFrame', 'Run this frame')}
            </Button>
        );
        // the bar's disabled states explain themselves under it
        const reason = video === undefined ? waitingText : cannotRunText;
        const waiting = reason !== undefined && !isFake && (
            <Text
                fontSize="xs"
                opacity={0.7}
            >
                {reason}
            </Text>
        );

        if (isWide) {
            return (
                <VStack
                    className="nodrag"
                    spacing={1}
                    w="full"
                >
                    {slider}
                    {/* the range group wraps onto its own line when the view is too narrow */}
                    <Flex
                        align="center"
                        columnGap={1}
                        flexWrap="wrap"
                        rowGap={2}
                        w="full"
                    >
                        <HStack spacing={1}>
                            {stepButtons}
                            {frameBox}
                        </HStack>
                        <Spacer />
                        {waiting}
                        <Flex
                            align="center"
                            columnGap={1}
                            flexWrap="wrap"
                            rowGap={2}
                        >
                            {companions && (
                                <Flex
                                    align="center"
                                    // each companion is a full input row; a number box keeps
                                    // room for six digits
                                    sx={{ '& > *': { flex: '0 0 auto', w: '13rem' } }}
                                >
                                    {companions}
                                </Flex>
                            )}
                            {rangeButtons}
                            {runButton}
                        </Flex>
                    </Flex>
                </VStack>
            );
        }

        return (
            <WithoutLabel>
                <VStack
                    className="nodrag"
                    outline="none"
                    spacing={1}
                    tabIndex={-1}
                    w="full"
                    onKeyDownCapture={onKeyDownCapture}
                >
                    {slider}
                    <HStack
                        spacing={0.5}
                        w="full"
                    >
                        {stepButtons}
                        <Spacer />
                        {rangeButtons}
                    </HStack>
                    <HStack w="full">{frameBox}</HStack>
                    {companions && (
                        // each companion brings its own container's padding
                        <Box
                            mx={-2}
                            w="calc(100% + 1rem)"
                        >
                            {companions}
                        </Box>
                    )}
                    {runButton}
                    {waiting}
                </VStack>
            </WithoutLabel>
        );
    }
);

export const VideoSeekInput = memo(
    ({
        value,
        setValue,
        input,
        isLocked,
        isConnected,
        nodeId,
        nodeSchemaId,
    }: InputProps<'video-seek', number>) => {
        const { def, label, framesInput, inInput, outInput } = input;
        const { t } = useTranslation();
        const { isFake } = useContext(FakeNodeContext);
        const { setNodeInputValue } = useContext(GlobalContext);
        const { getNode, getEdges } = useReactFlow<NodeData, EdgeData>();

        // The node's Frame Count and FPS as its last run reported them. A run makes them stale
        // (any input change, this one's too, runs the node again) but leaves them until the
        // next run reports its own. Each is selected alone: a new frame of the same video
        // renders the bar nothing again.
        const seekVideoOf = (types: OutputTypes | undefined) => getNodeSeekVideo(types, input);
        const frameCount = useContextSelector(GlobalVolatileContext, (c) =>
            nodeId === undefined || isFake
                ? undefined
                : seekVideoOf(c.outputDataMap.get(nodeId)?.types)?.frameCount
        );
        const fps = useContextSelector(GlobalVolatileContext, (c) =>
            nodeId === undefined || isFake
                ? undefined
                : seekVideoOf(c.outputDataMap.get(nodeId)?.types)?.fps
        );
        const video = useMemo(
            () => (frameCount === undefined || fps === undefined ? undefined : { frameCount, fps }),
            [frameCount, fps]
        );

        // A node with connections (Clip Frames behind its clip) has the Preview of a frame
        // rendered through the nodes before it, at that Seek on that frame alone (its own
        // Frames, Start and End stay); one without (Load Video) renders it by itself at that
        // Seek. The value is the gesture's end's. During a gesture the frames bring only the
        // previews of the node's views on screen; the end renders its frame with all of them,
        // alone (no move first: the end is no heavier than a move).
        const isConnectedNode = useCallback(
            () => nodeId !== undefined && getEdges().some((e) => e.target === nodeId),
            [nodeId, getEdges]
        );
        const overridesFor = useCallback(
            (n: number) => {
                if (nodeId === undefined || input.id === undefined) return undefined;
                const seek = { [input.id]: n };
                const overrides = isConnectedNode()
                    ? { ...oneFrameOverride(input, n), ...seek }
                    : seek;
                return new Map([[nodeId, overrides]]);
            },
            [nodeId, input, isConnectedNode]
        );
        const renderPreview = useContextSelector(ExecutionContext, (c) => c.renderPreview);
        const renderAlone = useContextSelector(ExecutionContext, (c) => c.renderAlone);
        const reserveAlone = useContextSelector(ExecutionContext, (c) => c.reserveAlone);
        const render = useCallback(
            (id: string, overrides: ReadonlyMap<string, InputData>, endsGesture: boolean) =>
                isConnectedNode()
                    ? renderPreview(id, overrides, !endsGesture)
                    : renderAlone(id, overrides.get(id) ?? {}, !endsGesture),
            [isConnectedNode, renderPreview, renderAlone]
        );
        // a gesture's end renders alone: a typed frame costs one run
        const endWithoutMove = useCallback(() => true, []);
        const previewOptions = useMemo(
            () => ({ render, kind: nodeSchemaId, endWithoutMove }),
            [render, nodeSchemaId, endWithoutMove]
        );
        const showPreview = useSeekPreview(nodeId, overridesFor, previewOptions);
        // A gesture shows its frames without setting the value; its end sets it (a node by
        // itself does not run again for it: the end's frame is the one rendered or asked for).
        const setSeek = useCallback(
            (n: number, startsGesture: boolean) => {
                if (isFake) {
                    setValue(n);
                } else {
                    showPreview(n, startsGesture);
                }
            },
            [isFake, setValue, showPreview]
        );
        const endSeek = useCallback(
            (n: number, isLeaving = false) => {
                if (isFake) {
                    setValue(n);
                    return;
                }
                // the bar can go away with its node (its chain closed): nothing is set then
                if (nodeId === undefined || getNode(nodeId) === undefined) return;
                // a bar going away renders nothing more through its queue
                const willRender = !isLeaving && showPreview(n, false, true);
                if (input.id !== undefined && !isConnectedNode()) {
                    reserveAlone(nodeId, { [input.id]: n }, willRender);
                }
                setValue(n);
                // A node with connections does not run by itself: its picture would keep the
                // gesture's previews alone. Its frame renders complete now.
                const complete = overridesFor(n);
                if (isLeaving && isConnectedNode() && complete !== undefined) {
                    renderPreview(nodeId, complete).catch(log.error);
                }
            },
            [
                nodeId,
                isFake,
                getNode,
                isConnectedNode,
                reserveAlone,
                input,
                setValue,
                showPreview,
                overridesFor,
                renderPreview,
            ]
        );

        // Set start and Set end set the node's frame range (Frames: Range) at the frame; one
        // past the other moves it along, so the range stays valid.
        const setRange = useCallback(
            (isIn: boolean, n: number) => {
                if (nodeId === undefined) return;
                const [inputId, otherId] = isIn ? [inInput, outInput] : [outInput, inInput];
                const other = getNode(nodeId)?.data.inputData[otherId];
                setNodeInputValue(nodeId, framesInput, FRAMES_RANGE);
                setNodeInputValue(nodeId, inputId, n);
                if (typeof other === 'number' && (isIn ? other < n : other > n)) {
                    setNodeInputValue(nodeId, otherId, n);
                }
            },
            [nodeId, inInput, outInput, framesInput, getNode, setNodeInputValue]
        );

        // what the node feeds runs on its frame (Save Video writes its one-frame picture), with
        // what that depends on; every other seek bar's node in it on its own
        const run = useRunThisFrame();
        const runThisFrame = useCallback(
            (n: number) => {
                if (nodeId === undefined) return;
                // the bar's frame first becomes the node's, as at a gesture's end
                setValue(n);
                run(new Map([[nodeId, n]]), { nodeId, kind: 'downstream' });
            },
            [nodeId, setValue, run]
        );

        const refactor = useInputRefactor(nodeId, input, value, isConnected);

        return (
            <SeekBar
                committed={value ?? def}
                def={def}
                endSeek={endSeek}
                isDisabled={isLocked || isConnected || video === undefined}
                label={label}
                menuItems={refactor}
                runThisFrame={runThisFrame}
                setRange={setRange}
                setSeek={setSeek}
                video={video}
                waitingText={t(
                    'inputs.videoSeek.waitingForVideo',
                    'The seek bar works once the node has read the video.'
                )}
            />
        );
    }
);
