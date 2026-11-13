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
import { EdgeData, InputId, NodeData } from '../../../common/common-types';
import { log } from '../../../common/log';
import { BackendContext } from '../../contexts/BackendContext';
import { ExecutionContext, ExecutionStatus } from '../../contexts/ExecutionContext';
import { FakeNodeContext } from '../../contexts/FakeExampleContext';
import { GlobalContext, GlobalVolatileContext } from '../../contexts/GlobalNodeState';
import { InputContext } from '../../contexts/InputContext';
import { parsePastedNumber } from '../../helpers/pastedNumber';
import { LINEAR_SCALE } from '../../helpers/sliderScale';
import {
    SeekStep,
    clampFrame,
    formatTimecode,
    getSeekVideo,
    numericOutputValue,
    scrubFrame,
    seekStepForKey,
    stepFrame,
} from '../../helpers/videoSeek';
import { useContextMenu } from '../../hooks/useContextMenu';
import { useInputRefactor } from '../../hooks/useInputRefactor';
import { useSeekPreview } from '../../hooks/useSeekPreview';
import { AdvancedNumberInput } from './elements/AdvanceNumberInput';
import { SliderStyle, StyledSlider } from './elements/StyledSlider';
import { WithoutLabel } from './InputContainer';
import { InputProps } from './props';

/** The value of the `framesInput` enum for a frame range (Start..End). */
const FRAMES_RANGE = 'range';
/** While the slider is dragged, its frame is set at most this often (ms), and on release. */
const DRAG_COMMIT_INTERVAL = 150;

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

export const VideoSeekInput = memo(
    ({
        value,
        setValue,
        input,
        isLocked,
        isConnected,
        nodeId,
    }: InputProps<'video-seek', number>) => {
        const { def, label, framesInput, inInput, outInput, frameCountOutput, fpsOutput } = input;
        const { t } = useTranslation();
        const { isFake } = useContext(FakeNodeContext);
        const { conditionallyInactive } = useContext(InputContext);
        const { isWide, companions } = useContext(SeekBarLayoutContext);
        const { setNodeInputValue } = useContext(GlobalContext);
        const { getNode, getEdges } = useReactFlow<NodeData, EdgeData>();
        const isBackendReady = useContextSelector(BackendContext, (c) => c.isBackendReady);
        const status = useContextSelector(ExecutionContext, (c) => c.status);
        const runWithOverrides = useContextSelector(ExecutionContext, (c) => c.runWithOverrides);

        // The node's Frame Count and FPS as its last run reported them. A run makes them stale
        // (any input change, this one's too, runs the node again) but leaves them until the
        // next run reports its own.
        const types = useContextSelector(GlobalVolatileContext, (c) =>
            nodeId === undefined || isFake ? undefined : c.outputDataMap.get(nodeId)?.types
        );
        const video = useMemo(
            () =>
                getSeekVideo(
                    numericOutputValue(types, frameCountOutput),
                    numericOutputValue(types, fpsOutput)
                ),
            [types, frameCountOutput, fpsOutput]
        );

        const isDisabled = isLocked || isConnected || video === undefined;

        // this run only: the node keeps its own Frames, Start and End
        const oneFrame = useCallback(
            (n: number) => ({ [framesInput]: FRAMES_RANGE, [inInput]: n, [outInput]: n }),
            [framesInput, inInput, outInput]
        );
        // A node without connections renders the Preview of each new frame by itself (Load
        // Video); one with connections (Clip Frames behind its clip) has it rendered here.
        const showPreview = useSeekPreview(nodeId, oneFrame);
        // startsGesture: a drag begins, a step, a typed frame (a drag's last frame is no new one)
        const setSeek = useCallback(
            (n: number, startsGesture: boolean) => {
                setValue(n);
                if (
                    nodeId !== undefined &&
                    !isFake &&
                    getEdges().some((e) => e.target === nodeId)
                ) {
                    showPreview(n, startsGesture);
                }
            },
            [setValue, nodeId, isFake, getEdges, showPreview]
        );

        // The frame shown: ahead of the value while the slider is dragged.
        const committed = value ?? def;
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
        const step = (seekStep: SeekStep) => {
            if (video === undefined) return;
            commit(stepFrame(seekStep, frame, video.frameCount, video.fps));
        };

        // A drag sets its frame on release, and every DRAG_COMMIT_INTERVAL ms while it moves.
        const lastDragCommitRef = useRef(0);
        const dragTimerRef = useRef<ReturnType<typeof setTimeout>>();
        // whether the next frame the drag sets is its first
        const dragStartsRef = useRef(false);
        useEffect(() => () => clearTimeout(dragTimerRef.current), []);
        const onSliderChange = useCallback(
            (n: number) => {
                if (!draggingRef.current) dragStartsRef.current = true;
                draggingRef.current = true;
                setFrame(n);
                clearTimeout(dragTimerRef.current);
                const setDragged = () => {
                    lastDragCommitRef.current = Date.now();
                    setSeek(n, dragStartsRef.current);
                    dragStartsRef.current = false;
                };
                const wait = DRAG_COMMIT_INTERVAL - (Date.now() - lastDragCommitRef.current);
                if (wait <= 0) {
                    setDragged();
                } else {
                    dragTimerRef.current = setTimeout(setDragged, wait);
                }
            },
            [setSeek]
        );
        const onSliderChangeEnd = useCallback(
            (n: number) => {
                clearTimeout(dragTimerRef.current);
                draggingRef.current = false;
                lastDragCommitRef.current = Date.now();
                commit(n, dragStartsRef.current);
                dragStartsRef.current = false;
            },
            [commit]
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
            step(seekStep);
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
                stepRef.current(seekStep);
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

        // Set start and Set end set the node's frame range (Frames: Range) at the frame; one
        // past the other moves it along, so the range stays valid.
        const setRangeEnd = (inputId: InputId, otherId: InputId, isIn: boolean) => {
            if (nodeId === undefined || video === undefined) return;
            const n = clampFrame(frame, video.frameCount);
            const other = getNode(nodeId)?.data.inputData[otherId];
            setNodeInputValue(nodeId, framesInput, FRAMES_RANGE);
            setNodeInputValue(nodeId, inputId, n);
            if (typeof other === 'number' && (isIn ? other < n : other > n)) {
                setNodeInputValue(nodeId, otherId, n);
            }
        };

        const canRun = isBackendReady && status === ExecutionStatus.READY;
        const runThisFrame = () => {
            if (nodeId === undefined || video === undefined) return;
            const n = clampFrame(frame, video.frameCount);
            runWithOverrides(new Map([[nodeId, oneFrame(n)]])).catch(log.error);
        };

        const refactor = useInputRefactor(nodeId, input, value, isConnected);
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
                                    commit(n);
                                }
                            })
                            .catch(log.error);
                    }}
                >
                    {t('inputs.number.paste', 'Paste')}
                </MenuItem>
                {refactor}
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
        const rangeButtons = (
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
                        onClick={() => setRangeEnd(inInput, outInput, true)}
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
                        onClick={() => setRangeEnd(outInput, inInput, false)}
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
                    setInput={commit}
                    setInputString={setFrameString}
                    onContextMenu={menu.onContextMenu}
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
                isDisabled={isDisabled || !canRun}
                leftIcon={<MdPlayArrow />}
                size="xs"
                variant="outline"
                w={isWide ? undefined : 'full'}
                onClick={runThisFrame}
            >
                {t('inputs.videoSeek.runThisFrame', 'Run this frame')}
            </Button>
        );
        const waiting = video === undefined && !isFake && (
            <Text
                fontSize="xs"
                opacity={0.7}
            >
                {t(
                    'inputs.videoSeek.waitingForVideo',
                    'The seek bar works once the node has read the video.'
                )}
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
