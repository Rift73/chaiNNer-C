import {
    Box,
    Button,
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
import { useContext, useContextSelector } from 'use-context-selector';
import { EdgeData, InputId, NodeData } from '../../../common/common-types';
import { log } from '../../../common/log';
import { BackendContext } from '../../contexts/BackendContext';
import { ExecutionContext, ExecutionStatus } from '../../contexts/ExecutionContext';
import { FakeNodeContext } from '../../contexts/FakeExampleContext';
import { GlobalContext, GlobalVolatileContext } from '../../contexts/GlobalNodeState';
import { InputContext } from '../../contexts/InputContext';
import { LINEAR_SCALE } from '../../helpers/sliderScale';
import {
    SeekStep,
    clampFrame,
    formatTimecode,
    numericOutputValue,
    stepFrame,
} from '../../helpers/videoSeek';
import { useContextMenu } from '../../hooks/useContextMenu';
import { useInputRefactor } from '../../hooks/useInputRefactor';
import { AdvancedNumberInput } from './elements/AdvanceNumberInput';
import { SliderStyle, StyledSlider } from './elements/StyledSlider';
import { WithoutLabel } from './InputContainer';
import { InputProps } from './props';

/** The value of the `framesInput` enum for a frame range (In..Out). */
const FRAMES_RANGE = 'range';
/** While the slider is dragged, its frame is set at most this often (ms), and on release. */
const DRAG_COMMIT_INTERVAL = 150;

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
        const { setNodeInputValue } = useContext(GlobalContext);
        const { getNode } = useReactFlow<NodeData, EdgeData>();
        const isBackendReady = useContextSelector(BackendContext, (c) => c.isBackendReady);
        const status = useContextSelector(ExecutionContext, (c) => c.status);
        const runWithOverrides = useContextSelector(ExecutionContext, (c) => c.runWithOverrides);

        // The node's Frame Count and FPS as its last run reported them. A run makes them stale
        // (any input change, this one's too, runs the node again) but leaves them until the
        // next run reports its own.
        const types = useContextSelector(GlobalVolatileContext, (c) =>
            nodeId === undefined || isFake ? undefined : c.outputDataMap.get(nodeId)?.types
        );
        const video = useMemo(() => {
            const frameCount = numericOutputValue(types, frameCountOutput);
            const fps = numericOutputValue(types, fpsOutput);
            if (frameCount === undefined || fps === undefined || frameCount < 1 || fps <= 0) {
                return undefined;
            }
            return { frameCount: Math.round(frameCount), fps };
        }, [types, frameCountOutput, fpsOutput]);

        const isDisabled = isLocked || isConnected || video === undefined;

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
            (n: number) => {
                if (video === undefined) return;
                const clamped = clampFrame(n, video.frameCount);
                setFrame(clamped);
                setValue(clamped);
            },
            [video, setValue]
        );
        const step = (seekStep: SeekStep) => {
            if (video === undefined) return;
            commit(stepFrame(seekStep, frame, video.frameCount, video.fps));
        };

        // A drag sets its frame on release, and every DRAG_COMMIT_INTERVAL ms while it moves.
        const lastDragCommitRef = useRef(0);
        const dragTimerRef = useRef<ReturnType<typeof setTimeout>>();
        useEffect(() => () => clearTimeout(dragTimerRef.current), []);
        const onSliderChange = useCallback(
            (n: number) => {
                draggingRef.current = true;
                setFrame(n);
                clearTimeout(dragTimerRef.current);
                const setDragged = () => {
                    lastDragCommitRef.current = Date.now();
                    setValue(n);
                };
                const wait = DRAG_COMMIT_INTERVAL - (Date.now() - lastDragCommitRef.current);
                if (wait <= 0) {
                    setDragged();
                } else {
                    dragTimerRef.current = setTimeout(setDragged, wait);
                }
            },
            [setValue]
        );
        const onSliderChangeEnd = useCallback(
            (n: number) => {
                clearTimeout(dragTimerRef.current);
                draggingRef.current = false;
                lastDragCommitRef.current = Date.now();
                commit(n);
            },
            [commit]
        );

        // Left/Right step a frame (a second with Shift), Home/End go to the first/last frame,
        // while the seek bar has the focus (but not its frame box, whose text they edit).
        const onKeyDownCapture = (event: React.KeyboardEvent) => {
            if (isDisabled || event.target instanceof HTMLInputElement) return;
            let seekStep: SeekStep;
            switch (event.key) {
                case 'ArrowLeft':
                    seekStep = event.shiftKey ? 'second-back' : 'frame-back';
                    break;
                case 'ArrowRight':
                    seekStep = event.shiftKey ? 'second-forward' : 'frame-forward';
                    break;
                case 'Home':
                    seekStep = 'first';
                    break;
                case 'End':
                    seekStep = 'last';
                    break;
                default:
                    return;
            }
            // the slider's own keys and the editor's (moving the node) do not apply
            event.preventDefault();
            event.stopPropagation();
            step(seekStep);
        };

        // In and Out set the node's frame range (Frames: Range) at the frame; one past the
        // other moves it along, so the range stays valid.
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
            // this run only: the node keeps its own Frames, In and Out
            const overrides = { [framesInput]: FRAMES_RANGE, [inInput]: n, [outInput]: n };
            runWithOverrides(new Map([[nodeId, overrides]])).catch(log.error);
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
                                const n = Number(clipboardValue);
                                if (!Number.isNaN(n)) {
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
        const timecode = video ? formatTimecode(frame, video.fps) : '-:--:--.---';
        const frameDigits = String(Math.max(0, lastFrame)).length;

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
                    <Box w="full">
                        <StyledSlider
                            def={def}
                            isDisabled={isDisabled || lastFrame < 1}
                            max={sliderMax}
                            min={0}
                            scale={LINEAR_SCALE}
                            step={1}
                            style={sliderStyle}
                            tooltip={`${frame} (${timecode})`}
                            value={Math.min(frame, sliderMax)}
                            onChange={onSliderChange}
                            onChangeEnd={onSliderChangeEnd}
                            onContextMenu={menu.onContextMenu}
                        />
                    </Box>
                    <HStack
                        spacing={0.5}
                        w="full"
                    >
                        <SeekButton
                            icon={<MdFirstPage />}
                            isDisabled={isDisabled}
                            label={t('inputs.videoSeek.firstFrame', 'First frame (Home)')}
                            onClick={() => step('first')}
                        />
                        <SeekButton
                            icon={<MdFastRewind />}
                            isDisabled={isDisabled}
                            label={t('inputs.videoSeek.secondBack', 'Back 1 second (Shift+Left)')}
                            onClick={() => step('second-back')}
                        />
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
                        <SeekButton
                            icon={<MdFastForward />}
                            isDisabled={isDisabled}
                            label={t(
                                'inputs.videoSeek.secondForward',
                                'Forward 1 second (Shift+Right)'
                            )}
                            onClick={() => step('second-forward')}
                        />
                        <SeekButton
                            icon={<MdLastPage />}
                            isDisabled={isDisabled}
                            label={t('inputs.videoSeek.lastFrame', 'Last frame (End)')}
                            onClick={() => step('last')}
                        />
                        <Spacer />
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
                                {t('inputs.videoSeek.in', 'In')}
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
                                {t('inputs.videoSeek.out', 'Out')}
                            </Button>
                        </Tooltip>
                    </HStack>
                    <HStack w="full">
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
                        <Text
                            fontFamily="mono"
                            fontSize="xs"
                            opacity={video ? undefined : 0.5}
                        >
                            {timecode}
                        </Text>
                    </HStack>
                    <Button
                        className="nodrag"
                        h="1.5rem"
                        isDisabled={isDisabled || !canRun}
                        leftIcon={<MdPlayArrow />}
                        size="xs"
                        variant="outline"
                        w="full"
                        onClick={runThisFrame}
                    >
                        {t('inputs.videoSeek.runThisFrame', 'Run this frame')}
                    </Button>
                    {video === undefined && !isFake && (
                        <Text
                            fontSize="xs"
                            opacity={0.7}
                        >
                            {t(
                                'inputs.videoSeek.waitingForVideo',
                                'The seek bar works once the node has read the video.'
                            )}
                        </Text>
                    )}
                </VStack>
            </WithoutLabel>
        );
    }
);
