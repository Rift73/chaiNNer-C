/* eslint-disable no-nested-ternary */
import { ViewOffIcon, WarningIcon } from '@chakra-ui/icons';
import {
    Box,
    Center,
    HStack,
    IconButton,
    Image,
    Spinner,
    Text,
    Tooltip,
    useDisclosure,
} from '@chakra-ui/react';
import { Resizable } from 're-resizable';
import { CSSProperties, memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { MdOpenInFull } from 'react-icons/md';
import { useContextSelector } from 'use-context-selector';
import { OutputId, Size } from '../../../common/common-types';
import { ExecutionContext } from '../../contexts/ExecutionContext';
import { GlobalVolatileContext } from '../../contexts/GlobalNodeState';
import {
    PreviewImage,
    clampSplit,
    containedWidth,
    emptyAreaColor,
    getComparePair,
    imageBackdrop,
    pickImage,
} from '../../helpers/imageCompare';
import { useDevicePixelRatio } from '../../hooks/useDevicePixelRatio';
import { useMemoArray } from '../../hooks/useMemo';
import { useSettings } from '../../hooks/useSettings';
import { DragHandleSVG } from '../CustomIcons';
import { BackgroundButton, CompareImage, ComparePicture, CompareToolbar } from './ImageCompareView';
import { MaximizedImage, MaximizedSide } from './MaximizedImage';
import { OutputProps, UseOutputData } from './props';

const IMAGE_PREVIEW_SIZE = 216;
const DEFAULT_SIZE: Readonly<Size> = {
    width: IMAGE_PREVIEW_SIZE,
    height: IMAGE_PREVIEW_SIZE,
};

interface LargeImageBroadcastData {
    readonly previews: readonly PreviewImage[];
    readonly width: number;
    readonly height: number;
    readonly channels: number;
}

interface ShownImage {
    readonly id: OutputId;
    readonly label: string;
    readonly data: UseOutputData<LargeImageBroadcastData>;
}

export const LargeImageOutput = memo(
    ({
        output,
        id,
        useOutputData,
        animated,
        size = DEFAULT_SIZE,
        setSize,
        seekBar,
        compare,
    }: OutputProps) => {
        const { t } = useTranslation();
        const maximized = useDisclosure();

        const [currentSize, setCurrentSize] = useState<Size>();

        const dpr = useDevicePixelRatio();
        const zoom = useContextSelector(GlobalVolatileContext, (c) => c.zoom);
        const realWidth = (currentSize ?? size).width * zoom * dpr;

        const ownData = useOutputData<LargeImageBroadcastData>(output.id);
        // a lookup of this node's broadcast data, called per image of the viewer
        const getOutputData = useOutputData;
        // the viewer's images that have data; one image shows as a single preview
        const shown: readonly ShownImage[] = compare
            ? compare.options
                  .map((option) => ({
                      ...option,
                      data: getOutputData<LargeImageBroadcastData>(option.id),
                  }))
                  .filter((image) => image.data.last !== undefined)
            : [];
        const pair =
            compare &&
            getComparePair(
                shown.map((image) => image.id),
                compare.choice
            );
        const shownA = pair && shown.find((image) => image.id === pair.a);
        const shownB = pair && shown.find((image) => image.id === pair.b);
        // with no image of the viewer there, its own output has no data either
        const main = shown.length > 0 ? (shownA ?? shown[0]).data : ownData;
        const { last } = main;
        const stale = main.stale || (shownB?.data.stale ?? false);
        const lastB = shownB?.data.last;

        const { snapToGrid, snapToGridAmount, imagePreviewBackground } = useSettings();

        const imgBgColor = 'var(--node-image-preview-bg)';
        const fontColor = 'var(--node-image-preview-color)';
        const emptyColor = emptyAreaColor(imagePreviewBackground, imgBgColor);

        const previewImage = last ? pickImage(last.previews, realWidth) : null;
        // The widths of the node's views on screen, for the previews a gesture's frames bring:
        // the node's own, and the maximized view's while it is open.
        const setViewWidth = useContextSelector(ExecutionContext, (c) => c.setViewWidth);
        const nodeView = `${output.id} node`;
        useEffect(() => {
            setViewWidth(id, nodeView, realWidth);
            return () => setViewWidth(id, nodeView, undefined);
        }, [setViewWidth, id, nodeView, realWidth]);
        const setMaximizedWidth = useCallback(
            (width: number | undefined) => setViewWidth(id, `${output.id} maximized`, width, true),
            [setViewWidth, id, output.id]
        );
        // under the open maximized view the node keeps its picture: the view covers it
        const coveredUrl = useRef<string>();
        if (!maximized.isOpen || coveredUrl.current === undefined) {
            coveredUrl.current = previewImage?.url;
        }
        // the maximized view shows the largest preview the node sent
        const largestImage = last ? pickImage(last.previews, Infinity) : null;
        const imageAlt = t(
            'outputs.largeImage.imagePreviewFailed',
            'Image preview failed to load, probably unsupported file type.'
        );
        const imageBackground = last ? imageBackdrop(imagePreviewBackground, last.channels) : '';
        const imageRenderingFor = (preview: PreviewImage, displayWidth: number) =>
            zoom > 1 && displayWidth > IMAGE_PREVIEW_SIZE && preview.width < displayWidth
                ? 'pixelated'
                : 'auto';

        // B is contained in A's box, so it is drawn at most as wide as A
        const realWidthB =
            last && lastB
                ? containedWidth(
                      { width: realWidth, height: (realWidth * last.height) / last.width },
                      lastB
                  )
                : 0;
        const previewImageB = lastB ? pickImage(lastB.previews, realWidthB) : null;
        const largestImageB = lastB ? pickImage(lastB.previews, Infinity) : null;
        const split = clampSplit(compare?.choice?.split);
        const setPair = (newPair: { a: OutputId; b: OutputId }) =>
            compare?.setChoice({ ...compare.choice, ...newPair });
        const setSplit = (newSplit: number) =>
            compare?.setChoice({ ...compare.choice, split: newSplit });
        const toSide = (
            image: ShownImage,
            data: LargeImageBroadcastData,
            preview: PreviewImage
        ): MaximizedSide => ({
            url: preview.url,
            label: image.label,
            width: data.width,
            height: data.height,
            backdrop: imageBackdrop(imagePreviewBackground, data.channels),
            previewWidth: preview.width,
        });
        const toCompareImage = (
            image: ShownImage,
            data: LargeImageBroadcastData,
            preview: PreviewImage,
            displayWidth: number
        ): CompareImage => ({
            ...toSide(image, data, preview),
            imageRendering: imageRenderingFor(preview, displayWidth),
        });

        const splitView =
            pair &&
            shownA &&
            shownB &&
            last &&
            lastB &&
            previewImage &&
            previewImageB &&
            largestImage &&
            largestImageB
                ? {
                      a: toCompareImage(shownA, last, previewImage, realWidth),
                      b: toCompareImage(shownB, lastB, previewImageB, realWidthB),
                      maximizedA: toSide(shownA, last, largestImage),
                      maximizedB: toSide(shownB, lastB, largestImageB),
                  }
                : undefined;

        const maximizeLabel = t('outputs.largeImage.maximize', 'Maximize');
        const maximizeButton = (overPicture: boolean) => (
            <Tooltip
                closeOnClick
                hasArrow
                borderRadius={8}
                label={maximizeLabel}
                openDelay={500}
            >
                <IconButton
                    aria-label={maximizeLabel}
                    bgColor={overPicture ? imgBgColor : undefined}
                    className="nodrag"
                    icon={<MdOpenInFull />}
                    opacity={overPicture ? 0.75 : undefined}
                    position={overPicture ? 'absolute' : undefined}
                    right={overPicture ? 1 : undefined}
                    size="xs"
                    top={overPicture ? 1 : undefined}
                    variant="ghost"
                    zIndex={overPicture ? 100 : undefined}
                    onClick={maximized.onOpen}
                />
            </Tooltip>
        );

        const [resizeRef, setResizeRef] = useState<Resizable | null>(null);

        const firstRender = useRef(false);
        useEffect(() => {
            if (!firstRender.current) {
                resizeRef?.updateSize({
                    width: size.width,
                    height: size.height,
                });
                firstRender.current = true;
            }
        }, [resizeRef, size]);

        const imageStyle = useMemo((): CSSProperties | undefined => {
            if (!last) return undefined;
            const s = currentSize ?? size;
            const sizeRatio = s.width / s.height;
            const imageRatio = last.width / last.height;
            return {
                aspectRatio: `${last.width} / ${last.height}`,
                width: sizeRatio < imageRatio ? `min(${last.width}px, 100%)` : undefined,
                height: sizeRatio < imageRatio ? undefined : `min(${last.height}px, 100%)`,
            };
        }, [currentSize, size, last]);

        return (
            <Center
                flexDirection="column"
                h="full"
                minH="2rem"
                overflow="hidden"
                verticalAlign="middle"
                w="full"
            >
                {pair && splitView && (
                    <HStack
                        mb={1}
                        spacing={1}
                        w="full"
                    >
                        <CompareToolbar
                            options={shown}
                            pair={pair}
                            size="xs"
                            onChange={setPair}
                        />
                        <BackgroundButton size="xs" />
                        {maximizeButton(false)}
                    </HStack>
                )}
                <Resizable
                    defaultSize={size}
                    enable={{
                        top: false,
                        right: false,
                        bottom: false,
                        left: false,
                        topRight: false,
                        bottomRight: true,
                        bottomLeft: false,
                        topLeft: false,
                    }}
                    grid={useMemoArray<[number, number]>(
                        snapToGrid ? [snapToGridAmount, snapToGridAmount] : [1, 1]
                    )}
                    handleComponent={{
                        bottomRight: (
                            <Center
                                className="nodrag"
                                cursor="nwse-resize"
                                h="full"
                                ml={-1}
                                mt={-1}
                                w="full"
                            >
                                <DragHandleSVG
                                    color="var(--fg-300)"
                                    opacity={0.75}
                                />
                            </Center>
                        ),
                    }}
                    maxHeight={1024}
                    maxWidth={1024}
                    minHeight={IMAGE_PREVIEW_SIZE}
                    minWidth={IMAGE_PREVIEW_SIZE}
                    ref={(r) => {
                        setResizeRef(r);
                    }}
                    scale={zoom}
                    onResize={(e, direction, ref, d) => {
                        setCurrentSize({
                            width: size.width + d.width,
                            height: size.height + d.height,
                        });
                    }}
                    onResizeStop={(e, direction, ref, d) => {
                        let baseWidth = size.width;
                        let baseHeight = size.height;

                        if (baseWidth < IMAGE_PREVIEW_SIZE) baseWidth = IMAGE_PREVIEW_SIZE;
                        if (baseHeight < IMAGE_PREVIEW_SIZE) baseHeight = IMAGE_PREVIEW_SIZE;

                        setSize({
                            width: baseWidth + d.width,
                            height: baseHeight + d.height,
                        });
                        setCurrentSize(undefined);
                    }}
                >
                    <Center
                        h="full"
                        w="full"
                    >
                        <Box
                            display={stale ? 'block' : 'none'}
                            h="full"
                            pointerEvents="none"
                            position="absolute"
                            w="full"
                            zIndex="99"
                        >
                            <HStack
                                alignContent="center"
                                alignItems="center"
                                bgColor={imgBgColor}
                                borderRadius="md"
                                margin={2}
                                opacity={0.75}
                                position="absolute"
                                px={2}
                                py={1}
                                spacing={1}
                                verticalAlign="middle"
                                w="auto"
                                zIndex="99"
                            >
                                <WarningIcon
                                    boxSize={3}
                                    color={fontColor}
                                />
                                <Text
                                    color={fontColor}
                                    fontSize="sm"
                                    fontWeight={500}
                                >
                                    {t('node.outdated')}
                                </Text>
                            </HStack>
                        </Box>
                        <Center
                            bgColor={last ? emptyColor : imgBgColor}
                            borderRadius="md"
                            h="full"
                            maxH="full"
                            maxW="full"
                            minH={`${animated ? size.height : IMAGE_PREVIEW_SIZE}px`}
                            minW={`${animated ? size.width : IMAGE_PREVIEW_SIZE}px`}
                            overflow="hidden"
                            w="full"
                        >
                            {splitView ? (
                                <Center
                                    display="flex"
                                    h="full"
                                    maxH="full"
                                    maxW="full"
                                    w="full"
                                >
                                    <ComparePicture
                                        a={splitView.a}
                                        alt={imageAlt}
                                        b={splitView.b}
                                        emptyColor={emptyColor}
                                        split={split}
                                        style={imageStyle ?? {}}
                                        onSplitEnd={setSplit}
                                    />
                                </Center>
                            ) : last && previewImage ? (
                                <Center
                                    display="flex"
                                    h="full"
                                    maxH="full"
                                    maxW="full"
                                    w="full"
                                >
                                    <Image
                                        alt={imageAlt}
                                        background={imageBackground}
                                        draggable={false}
                                        src={
                                            maximized.isOpen
                                                ? coveredUrl.current ?? previewImage.url
                                                : previewImage.url
                                        }
                                        style={imageStyle}
                                        sx={{
                                            imageRendering: imageRenderingFor(
                                                previewImage,
                                                realWidth
                                            ),
                                        }}
                                    />
                                    <Box
                                        position="absolute"
                                        right={8}
                                        top={1}
                                        zIndex={100}
                                    >
                                        <BackgroundButton
                                            bgColor={imgBgColor}
                                            size="xs"
                                        />
                                    </Box>
                                    {maximizeButton(true)}
                                </Center>
                            ) : animated ? (
                                // 10 steps per second, as the node's running spinner: each
                                // repainted frame makes the OS compositor interrupt GPU work.
                                <Spinner
                                    speed="1.2s"
                                    style={{ animationTimingFunction: 'steps(12)' }}
                                />
                            ) : (
                                <HStack>
                                    <ViewOffIcon />
                                    <Text
                                        fontSize="sm"
                                        lineHeight="0.5rem"
                                    >
                                        {t(
                                            'outputs.largeImage.imageNotAvailable',
                                            'Image not available.'
                                        )}
                                    </Text>
                                </HStack>
                            )}
                        </Center>
                    </Center>
                </Resizable>
                {last && largestImage && (
                    <MaximizedImage
                        alt={imageAlt}
                        background={imageBackground}
                        compare={
                            pair && splitView
                                ? {
                                      options: shown,
                                      pair,
                                      onChange: setPair,
                                      a: splitView.maximizedA,
                                      b: splitView.maximizedB,
                                      split,
                                      onSplitEnd: setSplit,
                                  }
                                : undefined
                        }
                        height={last.height}
                        isOpen={maximized.isOpen}
                        previewWidth={largestImage.width}
                        seekBar={seekBar}
                        url={largestImage.url}
                        width={last.width}
                        onClose={maximized.onClose}
                        onViewWidth={setMaximizedWidth}
                    />
                )}
            </Center>
        );
    }
);
