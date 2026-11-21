import {
    Box,
    Button,
    HStack,
    IconButton,
    Image,
    Modal,
    ModalContent,
    ModalOverlay,
    Spacer,
    Tooltip,
} from '@chakra-ui/react';
import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { MdClose, MdFitScreen, MdFullscreen, MdFullscreenExit } from 'react-icons/md';
import { useContext, useContextSelector } from 'use-context-selector';
import { Size } from '../../../common/common-types';
import { log } from '../../../common/log';
import { HotkeysContext } from '../../contexts/HotKeyContext';
import {
    ComparePair,
    containedWidth,
    emptyAreaColor,
    isPixelated,
    realSizePercent,
    realSizeView,
} from '../../helpers/imageCompare';
import { PAN_ZOOM_FIT, PanZoom, zoomAt } from '../../helpers/videoSeek';
import { useDevicePixelRatio } from '../../hooks/useDevicePixelRatio';
import { useSettings } from '../../hooks/useSettings';
import { SeekBarLayoutContext } from '../inputs/VideoSeekInput';
import { BackgroundButton, CompareImage, ComparePicture, CompareToolbar } from './ImageCompareView';
import { CompareOption } from './props';

/** The wheel's zoom per pixel of scroll: 100 px (one notch) zooms by about 16 %. */
const WHEEL_ZOOM = 0.0015;

interface ViewButtonProps {
    label: string;
    icon: React.ReactElement;
    onClick: () => void;
}
const ViewButton = memo(({ label, icon, onClick }: ViewButtonProps) => (
    <Tooltip
        closeOnClick
        hasArrow
        borderRadius={8}
        label={label}
        openDelay={500}
    >
        <IconButton
            aria-label={label}
            icon={icon}
            size="sm"
            variant="ghost"
            onClick={onClick}
        />
    </Tooltip>
));

/** An image of the split view, with the width of the preview it shows. */
export type MaximizedSide = Omit<CompareImage, 'imageRendering'> & {
    readonly previewWidth: number;
};

/** The split view of two images, in place of the one picture. */
export interface MaximizedCompare {
    readonly options: readonly CompareOption[];
    readonly pair: ComparePair;
    readonly onChange: (pair: ComparePair) => void;
    readonly a: MaximizedSide;
    readonly b: MaximizedSide;
    readonly split: number;
    readonly onSplitEnd: (split: number) => void;
}

export interface MaximizedImageProps {
    isOpen: boolean;
    onClose: () => void;
    url: string;
    /** The image's real size, and the width of the preview at `url`. */
    width: number;
    height: number;
    previewWidth: number;
    alt: string;
    background: string;
    /** The node's seek bar, drawn across the view under the picture. */
    seekBar?: JSX.Element;
    compare?: MaximizedCompare;
    /** Told the picture's width on the screen (px) while the view is open, undefined at its close. */
    onViewWidth?: (width: number | undefined) => void;
}

/**
 * A preview picture over the whole window: fit to it, the wheel zooms about the pointer, a drag
 * pans, a double click fits it again. F (or the button) toggles full screen; Esc closes. A split
 * view of two images zooms and pans both as one; a drag of its handle moves the split instead.
 */
export const MaximizedImage = memo(
    ({
        isOpen,
        onClose,
        url,
        width,
        height,
        previewWidth,
        alt,
        background,
        seekBar,
        compare,
        onViewWidth,
    }: MaximizedImageProps) => {
        const { t } = useTranslation();
        const { imagePreviewBackground } = useSettings();
        // the view's keys are its own: the app's hotkeys (cut, paste ...) wait while it is open
        const setHotkeysEnabled = useContextSelector(HotkeysContext, (c) => c.setHotkeysEnabled);
        useEffect(() => {
            if (!isOpen) return;
            setHotkeysEnabled(false);
            return () => setHotkeysEnabled(true);
        }, [isOpen, setHotkeysEnabled]);
        // the node's layout, with its Frames, Start and End, spread across the view
        const nodeLayout = useContext(SeekBarLayoutContext);
        const wideLayout = useMemo(() => ({ ...nodeLayout, isWide: true }), [nodeLayout]);
        const contentRef = useRef<HTMLElement>(null);
        const [view, setView] = useState<PanZoom>(PAN_ZOOM_FIT);
        const [isFullscreen, setIsFullscreen] = useState(false);
        const panRef = useRef<{ x: number; y: number; pointerId: number }>();

        useEffect(() => {
            if (!isOpen) setView(PAN_ZOOM_FIT);
        }, [isOpen]);
        useEffect(() => {
            const onChange = () => setIsFullscreen(document.fullscreenElement !== null);
            document.addEventListener('fullscreenchange', onChange);
            return () => document.removeEventListener('fullscreenchange', onChange);
        }, []);

        const toggleFullscreen = useCallback(() => {
            if (document.fullscreenElement) {
                document.exitFullscreen().catch(log.error);
            } else {
                contentRef.current?.requestFullscreen().catch(log.error);
            }
        }, []);
        const close = useCallback(() => {
            if (document.fullscreenElement) document.exitFullscreen().catch(log.error);
            onClose();
        }, [onClose]);

        const dpr = useDevicePixelRatio();
        // the picture's area, for the images' widths on the screen
        const [area, setArea] = useState<Size>();
        const areaObserver = useRef<ResizeObserver>();
        const areaRef = useCallback((element: HTMLDivElement | null) => {
            areaObserver.current?.disconnect();
            areaObserver.current = undefined;
            if (!element) return;
            const observer = new ResizeObserver(([entry]) => {
                const { width: areaWidth, height: areaHeight } = entry.contentRect;
                // an area not laid out yet has no fit
                setArea(
                    areaWidth > 0 && areaHeight > 0
                        ? { width: areaWidth, height: areaHeight }
                        : undefined
                );
            });
            observer.observe(element);
            areaObserver.current = observer;
        }, []);

        const transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
        // the widths at the fit: the main image (A of a split) fit to the area, B contained in A
        const main = compare?.a ?? { width, height };
        const fitWidth = area ? containedWidth(area, main) : 0;
        const shownWidth = fitWidth * view.scale * dpr;
        useEffect(() => {
            if (!isOpen || onViewWidth === undefined || shownWidth === 0) return undefined;
            onViewWidth(shownWidth);
            return () => onViewWidth(undefined);
        }, [isOpen, onViewWidth, shownWidth]);
        const fitWidthB = compare
            ? containedWidth(
                  { width: fitWidth, height: (fitWidth * main.height) / main.width },
                  compare.b
              )
            : 0;
        const renderingFor = (shownFitWidth: number, shownPreviewWidth: number) =>
            area && isPixelated(shownFitWidth * view.scale, shownPreviewWidth, dpr)
                ? 'pixelated'
                : 'auto';
        const zoomPercent = area
            ? realSizePercent(fitWidth * view.scale, main.width, dpr)
            : undefined;
        // the preview shown can be narrower than the image (none at full size above 8192 px):
        // the readout says that its real size enlarges the preview
        const mainPreviewWidth = compare?.a.previewWidth ?? previewWidth;

        const onKeyDown = (event: React.KeyboardEvent) => {
            if (event.target instanceof HTMLInputElement) return;
            if (event.key === 'f' || event.key === 'F') {
                event.preventDefault();
                toggleFullscreen();
            }
        };

        const onWheel = (event: React.WheelEvent<HTMLDivElement>) => {
            const rect = event.currentTarget.getBoundingClientRect();
            const px = event.clientX - rect.left - rect.width / 2;
            const py = event.clientY - rect.top - rect.height / 2;
            const factor = Math.exp(-event.deltaY * WHEEL_ZOOM);
            setView((v) => zoomAt(v, factor, px, py));
        };
        // an Alt-drag scrubs the video instead (the seek bar's own listener)
        const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
            if (event.button !== 0 || event.altKey) return;
            event.currentTarget.setPointerCapture(event.pointerId);
            panRef.current = { x: event.clientX, y: event.clientY, pointerId: event.pointerId };
        };
        const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
            const start = panRef.current;
            if (start?.pointerId !== event.pointerId) return;
            const dx = event.clientX - start.x;
            const dy = event.clientY - start.y;
            panRef.current = { ...start, x: event.clientX, y: event.clientY };
            setView((v) => (v.scale === 1 ? v : { ...v, x: v.x + dx, y: v.y + dy }));
        };
        const onPointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
            if (panRef.current?.pointerId === event.pointerId) panRef.current = undefined;
        };

        return (
            <Modal
                isOpen={isOpen}
                size="full"
                onClose={close}
            >
                <ModalOverlay />
                <ModalContent
                    bg="var(--window-bg)"
                    borderRadius={0}
                    // nokey: the editor behind does not move or delete its nodes on our keys
                    className="nodrag nokey"
                    display="flex"
                    flexDirection="column"
                    h="100vh"
                    m={0}
                    ref={contentRef}
                    onKeyDown={onKeyDown}
                >
                    <HStack
                        px={2}
                        py={1}
                    >
                        {compare && (
                            <CompareToolbar
                                options={compare.options}
                                pair={compare.pair}
                                size="sm"
                                onChange={compare.onChange}
                            />
                        )}
                        <Spacer />
                        {zoomPercent !== undefined && (
                            <Tooltip
                                closeOnClick
                                hasArrow
                                borderRadius={8}
                                label={t('outputs.largeImage.realSize', 'Real size')}
                                openDelay={500}
                            >
                                <Button
                                    size="sm"
                                    sx={{ fontVariantNumeric: 'tabular-nums' }}
                                    variant="ghost"
                                    onClick={() =>
                                        setView((v) => realSizeView(v, fitWidth, main.width, dpr))
                                    }
                                >
                                    {mainPreviewWidth < main.width
                                        ? t(
                                              'outputs.largeImage.zoomOfPreview',
                                              '{{percent}} % (preview {{width}} px)',
                                              { percent: zoomPercent, width: mainPreviewWidth }
                                          )
                                        : `${zoomPercent} %`}
                                </Button>
                            </Tooltip>
                        )}
                        <BackgroundButton size="sm" />
                        <ViewButton
                            icon={<MdFitScreen />}
                            label={t('outputs.largeImage.fit', 'Fit to the window (double click)')}
                            onClick={() => setView(PAN_ZOOM_FIT)}
                        />
                        <ViewButton
                            icon={isFullscreen ? <MdFullscreenExit /> : <MdFullscreen />}
                            label={t('outputs.largeImage.fullscreen', 'Full screen (F)')}
                            onClick={toggleFullscreen}
                        />
                        <ViewButton
                            icon={<MdClose />}
                            label={t('outputs.largeImage.close', 'Close (Esc)')}
                            onClick={close}
                        />
                    </HStack>
                    <Box
                        data-seek-scrub
                        bgColor={emptyAreaColor(imagePreviewBackground, 'var(--window-bg)')}
                        cursor={view.scale > 1 ? 'grab' : undefined}
                        flex={1}
                        minH={0}
                        overflow="hidden"
                        position="relative"
                        ref={areaRef}
                        onDoubleClick={() => setView(PAN_ZOOM_FIT)}
                        onPointerCancel={onPointerUp}
                        onPointerDown={onPointerDown}
                        onPointerMove={onPointerMove}
                        onPointerUp={onPointerUp}
                        onWheel={onWheel}
                    >
                        {compare ? (
                            <Box
                                alignItems="center"
                                display="flex"
                                h="full"
                                justifyContent="center"
                                style={{ transform, containerType: 'size' }}
                                w="full"
                            >
                                <ComparePicture
                                    a={{
                                        ...compare.a,
                                        imageRendering: renderingFor(
                                            fitWidth,
                                            compare.a.previewWidth
                                        ),
                                    }}
                                    alt={alt}
                                    b={{
                                        ...compare.b,
                                        imageRendering: renderingFor(
                                            fitWidthB,
                                            compare.b.previewWidth
                                        ),
                                    }}
                                    emptyColor={emptyAreaColor(
                                        imagePreviewBackground,
                                        'var(--window-bg)'
                                    )}
                                    split={compare.split}
                                    style={{
                                        aspectRatio: `${compare.a.width} / ${compare.a.height}`,
                                        width: `min(100cqw, 100cqh * ${
                                            compare.a.width / compare.a.height
                                        })`,
                                    }}
                                    onSplitEnd={compare.onSplitEnd}
                                />
                            </Box>
                        ) : (
                            <Image
                                alt={alt}
                                background={background}
                                draggable={false}
                                h="full"
                                objectFit="contain"
                                src={url}
                                style={{
                                    transform,
                                    imageRendering: renderingFor(fitWidth, previewWidth),
                                }}
                                w="full"
                            />
                        )}
                    </Box>
                    {seekBar && (
                        <Box
                            px={4}
                            py={2}
                            // the view connects nothing: no handles on its inputs
                            sx={{ '.react-flow__handle': { display: 'none' } }}
                        >
                            <SeekBarLayoutContext.Provider value={wideLayout}>
                                {seekBar}
                            </SeekBarLayoutContext.Provider>
                        </Box>
                    )}
                </ModalContent>
            </Modal>
        );
    }
);
