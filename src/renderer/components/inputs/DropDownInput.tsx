import { memo, useCallback } from 'react';
import { useValidDropDownValue } from '../../hooks/useValidDropDownValue';
import { AnchorSelector } from './elements/AnchorSelector';
import { Checkbox } from './elements/Checkbox';
import { DropDown } from './elements/Dropdown';
import { IconList } from './elements/IconList';
import { TabList } from './elements/TabList';
import { AutoLabel, InlineLabel, WithoutLabel } from './InputContainer';
import { InputProps } from './props';

type DropDownInputProps = InputProps<'dropdown', string | number>;

export const DropDownInput = memo(
    ({ value, setValue, input, isLocked, testCondition }: DropDownInputProps) => {
        const { options, def, label, preferredStyle, groups, hint, description } = input;

        // eslint-disable-next-line no-param-reassign
        value = useValidDropDownValue(value, setValue, input);

        const reset = useCallback(() => setValue(def), [setValue, def]);

        if (preferredStyle === 'checkbox' && options.length === 2) {
            // checkbox assumes the first options means yes and the second option means no
            return (
                <WithoutLabel>
                    <Checkbox
                        description={hint ? description : undefined}
                        isDisabled={isLocked}
                        label={label}
                        no={options[1]}
                        value={value}
                        yes={options[0]}
                        onChange={setValue}
                    />
                </WithoutLabel>
            );
        }

        if (preferredStyle === 'tabs') {
            return (
                <WithoutLabel>
                    <TabList
                        isDisabled={isLocked}
                        options={input.options}
                        value={value}
                        onChange={setValue}
                    />
                </WithoutLabel>
            );
        }

        if (preferredStyle === 'icons') {
            return (
                <InlineLabel input={input}>
                    <IconList
                        isDisabled={isLocked}
                        options={input.options}
                        value={value}
                        onChange={setValue}
                    />
                </InlineLabel>
            );
        }

        if (preferredStyle === 'anchor' && options.length === 9) {
            return (
                <InlineLabel input={input}>
                    <AnchorSelector
                        isDisabled={isLocked}
                        options={input.options}
                        value={value}
                        onChange={setValue}
                    />
                </InlineLabel>
            );
        }

        return (
            <AutoLabel input={input}>
                <DropDown
                    groups={groups}
                    isDisabled={isLocked}
                    options={input.options}
                    reset={reset}
                    testCondition={testCondition}
                    value={value}
                    onChange={setValue}
                />
            </AutoLabel>
        );
    }
);
