import {
    Box,
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
import { log } from '../../../common/log';
import { HotkeysContext } from '../../contexts/HotKeyContext';
import { PAN_ZOOM_FIT, PanZoom, zoomAt } from '../../helpers/videoSeek';
import { SeekBarLayoutContext } from '../inputs/VideoSeekInput';

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

export interface MaximizedImageProps {
    isOpen: boolean;
    onClose: () => void;
    url: string;
    alt: string;
    background: string;
    /** The node's seek bar, drawn across the view under the picture. */
    seekBar?: JSX.Element;
}

/**
 * A preview picture over the whole window: fit to it, the wheel zooms about the pointer, a drag
 * pans, a double click fits it again. F (or the button) toggles full screen; Esc closes.
 */
export const MaximizedImage = memo(
    ({ isOpen, onClose, url, alt, background, seekBar }: MaximizedImageProps) => {
        const { t } = useTranslation();
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
                        <Spacer />
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
                        cursor={view.scale > 1 ? 'grab' : undefined}
                        flex={1}
                        minH={0}
                        overflow="hidden"
                        position="relative"
                        onDoubleClick={() => setView(PAN_ZOOM_FIT)}
                        onPointerCancel={onPointerUp}
                        onPointerDown={onPointerDown}
                        onPointerMove={onPointerMove}
                        onPointerUp={onPointerUp}
                        onWheel={onWheel}
                    >
                        <Image
                            alt={alt}
                            background={background}
                            draggable={false}
                            h="full"
                            objectFit="contain"
                            src={url}
                            style={{
                                transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
                                imageRendering: view.scale > 2 ? 'pixelated' : 'auto',
                            }}
                            w="full"
                        />
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
