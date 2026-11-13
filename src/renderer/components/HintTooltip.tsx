import { SystemStyleObject, Tooltip, TooltipProps } from '@chakra-ui/react';
import { PointerEvent, ReactElement, cloneElement, memo, useEffect, useRef, useState } from 'react';
import { Markdown } from './Markdown';

// The tooltip's background is dark in the light theme and light in the dark theme, so inline code
// gets a neutral tint that reads on both instead of Chakra's light-theme code background.
const hintStyle: SystemStyleObject = {
    '& p': { my: 1.5 },
    '& code': { bg: 'rgba(128 128 128 / 35%)', overflowWrap: 'anywhere' },
};

const OPEN_DELAY = 500;

type PointerHandler = (event: PointerEvent) => void;
interface TriggerProps {
    onPointerEnter?: PointerHandler;
    onPointerLeave?: PointerHandler;
    onPointerDown?: PointerHandler;
}

export interface HintTooltipProps {
    /** Markdown. Without it, the child is rendered without a pop-up. */
    description: string | null | undefined;
    /**
     * The default opens beside the child, so a full-width child puts the pop-up beside the node
     * instead of over the inputs below.
     */
    placement?: TooltipProps['placement'];
    isDisabled?: boolean;
    /** Overrides the pointer, e.g. for an option highlighted by the keyboard. */
    isOpen?: boolean;
    children: ReactElement<TriggerProps>;
}

/**
 * A pop-up that opens only from the pointer resting on its child: Chakra's own tooltip also opens
 * on keyboard focus, which kept it open over the node after a dropdown pick returned the focus.
 * It closes on pointer down.
 */
export const HintTooltip = memo(
    ({
        description,
        placement = 'right-start',
        isDisabled = false,
        isOpen,
        children,
    }: HintTooltipProps) => {
        const [isHovered, setIsHovered] = useState(false);
        const timer = useRef<number>();
        useEffect(() => () => window.clearTimeout(timer.current), []);

        if (!description) return children;

        // the child's own handlers, which the clone below replaces
        const own: TriggerProps = children.props;
        const trigger = cloneElement(children, {
            onPointerEnter: (event: PointerEvent) => {
                own.onPointerEnter?.(event);
                window.clearTimeout(timer.current);
                timer.current = window.setTimeout(() => setIsHovered(true), OPEN_DELAY);
            },
            onPointerLeave: (event: PointerEvent) => {
                own.onPointerLeave?.(event);
                window.clearTimeout(timer.current);
                setIsHovered(false);
            },
            onPointerDown: (event: PointerEvent) => {
                own.onPointerDown?.(event);
                window.clearTimeout(timer.current);
                setIsHovered(false);
            },
        });

        return (
            <Tooltip
                hasArrow
                borderRadius={8}
                isOpen={!isDisabled && (isOpen ?? isHovered)}
                label={<Markdown nonInteractive>{description}</Markdown>}
                maxW="sm"
                placement={placement}
                px={2}
                py={1}
                sx={hintStyle}
            >
                {trigger}
            </Tooltip>
        );
    }
);
