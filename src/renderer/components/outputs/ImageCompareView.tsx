import { Box, HStack, IconButton, Image, Select, Text, Tooltip } from '@chakra-ui/react';
import React, { CSSProperties, memo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { MdContrast, MdSwapHoriz } from 'react-icons/md';
import { OutputId } from '../../../common/common-types';
import {
    ComparePair,
    chooseCompareSide,
    nextPreviewBackground,
    splitAt,
    splitForKey,
} from '../../helpers/imageCompare';
import { useMutSetting } from '../../hooks/useSettings';
import { CompareOption } from './props';

interface CompareToolbarProps {
    options: readonly CompareOption[];
    pair: ComparePair;
    onChange: (pair: ComparePair) => void;
    size: 'xs' | 'sm';
}

/** The pickers of the images on the two sides of the split, with a swap between them. */
export const CompareToolbar = memo(({ options, pair, onChange, size }: CompareToolbarProps) => {
    const { t } = useTranslation();
    const picker = (side: keyof ComparePair, label: string) => (
        <Select
            aria-label={label}
            className="nodrag nokey"
            minW={0}
            size={size}
            sx={{ textOverflow: 'ellipsis' }}
            title={`${label}: ${options.find((option) => option.id === pair[side])?.label ?? ''}`}
            value={pair[side]}
            onChange={(event) =>
                onChange(chooseCompareSide(pair, side, Number(event.target.value) as OutputId))
            }
        >
            {options.map((option) => (
                <option
                    key={option.id}
                    title={option.label}
                    value={option.id}
                >
                    {side === 'a' ? 'A' : 'B'} · {option.label}
                </option>
            ))}
        </Select>
    );
    const swapLabel = t('outputs.largeImage.swap', 'Swap A and B');
    return (
        <HStack
            flex={1}
            minW={0}
            spacing={1}
        >
            {picker('a', t('outputs.largeImage.compareA', 'Image on the left of the split (A)'))}
            <Tooltip
                closeOnClick
                hasArrow
                borderRadius={8}
                label={swapLabel}
                openDelay={500}
            >
                <IconButton
                    aria-label={swapLabel}
                    className="nodrag"
                    icon={<MdSwapHoriz />}
                    size={size}
                    variant="ghost"
                    onClick={() => onChange({ a: pair.b, b: pair.a })}
                />
            </Tooltip>
            {picker('b', t('outputs.largeImage.compareB', 'Image on the right of the split (B)'))}
        </HStack>
    );
});

interface BackgroundButtonProps {
    size: 'xs' | 'sm';
    /** The button's own background, for a button drawn over the picture. */
    bgColor?: string;
}

/** Cycles the empty area of every image preview through the theme colour, white and black. */
export const BackgroundButton = memo(({ size, bgColor }: BackgroundButtonProps) => {
    const { t } = useTranslation();
    const [background, setBackground] = useMutSetting('imagePreviewBackground');
    const labels = {
        theme: t('outputs.largeImage.backgroundTheme', 'Background: theme colour'),
        white: t('outputs.largeImage.backgroundWhite', 'Background: white'),
        black: t('outputs.largeImage.backgroundBlack', 'Background: black'),
    };
    return (
        <Tooltip
            closeOnClick
            hasArrow
            borderRadius={8}
            label={labels[background]}
            openDelay={500}
        >
            <IconButton
                aria-label={labels[background]}
                bgColor={bgColor}
                className="nodrag"
                icon={<MdContrast />}
                opacity={bgColor ? 0.75 : undefined}
                size={size}
                variant="ghost"
                onClick={() => setBackground(nextPreviewBackground)}
            />
        </Tooltip>
    );
});

export interface CompareImage {
    readonly url: string;
    readonly label: string;
    readonly width: number;
    readonly height: number;
    /** What is drawn behind the image's own pixels. */
    readonly backdrop: string;
    readonly imageRendering: CSSProperties['imageRendering'];
}

interface ComparePictureProps {
    a: CompareImage;
    b: CompareImage;
    alt: string;
    /** The colour where B, contained in A's box, leaves the box empty. */
    emptyColor: string;
    split: number;
    onSplitEnd: (split: number) => void;
    /** The size of the common box; its aspect ratio is A's. */
    style: CSSProperties;
}

const Chip = memo(({ text, side }: { text: string; side: 'left' | 'right' }) => (
    <Text
        bgColor="blackAlpha.600"
        borderRadius="sm"
        color="white"
        fontSize="xs"
        left={side === 'left' ? 1 : undefined}
        lineHeight="short"
        maxW="45%"
        noOfLines={1}
        pointerEvents="none"
        position="absolute"
        px={1.5}
        right={side === 'right' ? 1 : undefined}
        top={1}
    >
        {text}
    </Text>
));

/**
 * A on the left of a split and B on the right, both in one box with A's aspect ratio; B is
 * contained in it, so images of different sizes stay aligned. Dragging the handle or the Left and
 * Right keys move the split; it is kept when the drag ends.
 */
export const ComparePicture = memo(
    ({ a, b, alt, emptyColor, split, onSplitEnd, style }: ComparePictureProps) => {
        const { t } = useTranslation();
        const boxRef = useRef<HTMLDivElement>(null);
        const [dragSplit, setDragSplit] = useState<number>();
        const shown = dragSplit ?? split;

        const moveTo = (clientX: number) => {
            const rect = boxRef.current?.getBoundingClientRect();
            if (rect) setDragSplit(splitAt(clientX, rect.left, rect.width));
        };
        const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
            if (event.button !== 0) return;
            // the drag moves the split only: not the node, nor the maximized view's picture
            event.stopPropagation();
            event.currentTarget.setPointerCapture(event.pointerId);
            moveTo(event.clientX);
        };
        const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
            if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
            moveTo(event.clientX);
        };
        const onPointerEnd = () => {
            if (dragSplit === undefined) return;
            onSplitEnd(dragSplit);
            setDragSplit(undefined);
        };
        const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
            const next = splitForKey(shown, event.key);
            if (next === undefined) return;
            event.preventDefault();
            event.stopPropagation();
            onSplitEnd(next);
        };

        return (
            <Box
                flexShrink={0}
                position="relative"
                ref={boxRef}
                style={style}
            >
                <Image
                    alt={alt}
                    background={a.backdrop}
                    draggable={false}
                    h="full"
                    objectFit="contain"
                    src={a.url}
                    sx={{ imageRendering: a.imageRendering }}
                    w="full"
                />
                <Box
                    alignItems="center"
                    bgColor={emptyColor}
                    display="flex"
                    inset={0}
                    justifyContent="center"
                    position="absolute"
                    style={{
                        clipPath: `inset(0 0 0 ${shown * 100}%)`,
                        containerType: 'size',
                    }}
                >
                    <Image
                        alt={alt}
                        background={b.backdrop}
                        draggable={false}
                        src={b.url}
                        style={{
                            aspectRatio: `${b.width} / ${b.height}`,
                            width: `min(100cqw, 100cqh * ${b.width / b.height})`,
                        }}
                        sx={{ imageRendering: b.imageRendering }}
                    />
                </Box>
                <Chip
                    side="left"
                    text={`A · ${a.label}`}
                />
                <Chip
                    side="right"
                    text={`B · ${b.label}`}
                />
                <Box
                    aria-label={t('outputs.largeImage.split', 'Split between A and B')}
                    aria-valuemax={100}
                    aria-valuemin={0}
                    aria-valuenow={Math.round(shown * 100)}
                    bottom={0}
                    className="nodrag nokey"
                    cursor="ew-resize"
                    left={`${shown * 100}%`}
                    position="absolute"
                    role="slider"
                    tabIndex={0}
                    top={0}
                    transform="translateX(-50%)"
                    w="16px"
                    onDoubleClick={(event) => event.stopPropagation()}
                    onKeyDown={onKeyDown}
                    onLostPointerCapture={onPointerEnd}
                    onPointerDown={onPointerDown}
                    onPointerMove={onPointerMove}
                >
                    <Box
                        bgColor="white"
                        bottom={0}
                        boxShadow="0 0 2px black"
                        left="7px"
                        position="absolute"
                        top={0}
                        w="2px"
                    />
                    <Box
                        bgColor="white"
                        border="2px solid"
                        borderColor="blackAlpha.600"
                        borderRadius="full"
                        h="16px"
                        left={0}
                        position="absolute"
                        top="calc(50% - 8px)"
                        w="16px"
                    />
                </Box>
            </Box>
        );
    }
);
