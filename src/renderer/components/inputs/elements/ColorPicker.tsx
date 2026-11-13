import { Box } from '@chakra-ui/react';
import { memo, useMemo, useRef } from 'react';
import { HsvColorPicker as BuggedHsvColorPicker } from 'react-colorful';
import { HsvColor } from '../../../helpers/color';

interface PickerProps<T> {
    color: T;
    onChange: (value: T) => void;
}

export const HsvColorPicker = memo(({ color, onChange }: PickerProps<HsvColor>) => {
    // react-colorful rounds its internal HSV values, so it can change the given color as soon as
    // it shows it. Only onChange calls that the user's interaction causes are passed on.

    const pickerRef = useRef<HTMLDivElement>(null);
    const pickerLastSet = useRef(0);

    // eslint-disable-next-line react-hooks/exhaustive-deps
    const lastColorChange = useMemo(Date.now, [color.id]);

    return (
        <Box
            ref={pickerRef}
            w="full"
        >
            <BuggedHsvColorPicker
                color={color}
                onChange={(value) => {
                    const now = Date.now();
                    const sinceLastChange = now - lastColorChange;
                    const sinceLastSet = now - pickerLastSet.current;
                    if (
                        pickerRef.current?.matches(':focus-within') &&
                        (sinceLastSet < 200 || sinceLastChange > 200)
                    ) {
                        pickerLastSet.current = Date.now();
                        onChange(HsvColor.from(value));
                    }
                }}
            />
        </Box>
    );
});
