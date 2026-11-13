import { QuestionIcon } from '@chakra-ui/icons';
import { Box, Checkbox as ChakraCheckbox } from '@chakra-ui/react';
import { memo } from 'react';
import { DropDownInput, InputSchemaValue } from '../../../../common/common-types';
import { HintTooltip } from '../../HintTooltip';
import './Checkbox.scss';

type ArrayItem<T> = T extends readonly (infer I)[] ? I : never;
type Option = ArrayItem<DropDownInput['options']>;

export interface CheckboxProps {
    value: InputSchemaValue;
    onChange: (value: InputSchemaValue) => void;
    isDisabled?: boolean;
    yes: Option;
    no: Option;
    label: string;
    /** Markdown; the whole row opens it as a pop-up. */
    description?: string;
}

export const Checkbox = memo(
    ({ value, onChange, isDisabled, yes, no, label, description }: CheckboxProps) => {
        const checkbox = (
            <ChakraCheckbox
                className="chainner-node-checkbox"
                colorScheme="gray"
                isChecked={value === yes.value}
                isDisabled={isDisabled}
                onChange={(e) => {
                    const selected = e.target.checked ? yes : no;
                    onChange(selected.value);
                }}
            >
                <Box
                    as="span"
                    fontSize="14px"
                    verticalAlign="text-top"
                >
                    {label}
                </Box>
                {description && (
                    <QuestionIcon
                        boxSize={3}
                        ml={1}
                        verticalAlign="baseline"
                    />
                )}
            </ChakraCheckbox>
        );
        if (!description) return checkbox;

        // the checkbox forwards its ref to the hidden input, so the pop-up anchors on a full-width
        // wrapper, which also puts it beside the node
        return (
            <HintTooltip description={description}>
                <Box>{checkbox}</Box>
            </HintTooltip>
        );
    }
);
