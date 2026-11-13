import { ChevronDownIcon, SearchIcon } from '@chakra-ui/icons';
import {
    Box,
    Button,
    Divider,
    Input,
    InputGroup,
    InputLeftElement,
    Popover,
    PopoverContent,
    PopoverTrigger,
    Portal,
    Select,
    Text,
    useColorModeValue,
    useDisclosure,
} from '@chakra-ui/react';
import {
    ChangeEvent,
    KeyboardEvent,
    RefObject,
    memo,
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
} from 'react';
import { useTranslation } from 'react-i18next';
import {
    Condition,
    DropDownInput,
    DropdownGroup,
    InputSchemaValue,
} from '../../../../common/common-types';
import {
    DropDownItem,
    DropDownOptionItem,
    filterDropDownItems,
    getDropDownItems,
    getDropDownKind,
    getUnavailableMask,
} from '../../../helpers/dropDownItems';
import { HintTooltip } from '../../HintTooltip';

export interface DropDownProps {
    value: InputSchemaValue;
    onChange: (value: InputSchemaValue) => void;
    reset: () => void;
    isDisabled?: boolean;
    options: DropDownInput['options'];
    groups?: readonly DropdownGroup[];
    testCondition?: (condition: Condition) => boolean;
}

const isUnavailable = (unavailable: string, index: number): boolean => unavailable[index] === '1';

const renderSelectItem = (item: DropDownItem, unavailable: string): JSX.Element[] => {
    const { index } = item;
    if (item.kind === 'option') {
        return [
            <option
                disabled={isUnavailable(unavailable, index)}
                key={index}
                style={{ fontSize: '120%' }}
                value={index}
            >
                {item.option.option}
            </option>,
        ];
    }

    const { label } = item.group;
    const pseudoValue = `--category-${index}-${label ?? ''}`;
    if (label) {
        return [
            <option
                disabled
                key={pseudoValue}
                style={{
                    fontSize: '110%',
                    fontWeight: 'bold',
                    textAlign: 'center',
                }}
                value={pseudoValue}
            >
                {label}
            </option>,
        ];
    }
    return [
        <option
            disabled
            key={`--hr-${pseudoValue}`}
            style={{
                fontSize: '85%',
                color: 'rgba(128 128 128 / 70%)',
            }}
            value={`--hr-${pseudoValue}`}
        >
            {'⎯'.repeat(30)}
        </option>,
    ];
};

interface ListProps {
    items: readonly DropDownItem[];
    options: DropDownInput['options'];
    /** See {@link getUnavailableMask}. */
    unavailable: string;
    selection: number;
    isDisabled?: boolean;
    onChange: (value: InputSchemaValue) => void;
}

interface SelectDropDownProps extends ListProps {
    reset: () => void;
}
const SelectDropDown = memo(
    ({
        items,
        options,
        unavailable,
        selection,
        isDisabled,
        onChange,
        reset,
    }: SelectDropDownProps) => {
        const optionElements = useMemo(
            () => items.flatMap((item) => renderSelectItem(item, unavailable)),
            [items, unavailable]
        );

        const handleChange = (event: ChangeEvent<HTMLSelectElement>) => {
            const selectedIndex = Number(event.target.value);
            const selectedValue = options[selectedIndex]?.value as InputSchemaValue | undefined;
            if (selectedValue === undefined) {
                reset();
            } else {
                onChange(selectedValue);
            }
        };

        return (
            <Select
                borderRadius="lg"
                className="nodrag"
                disabled={isDisabled}
                draggable={false}
                size="sm"
                style={{ contain: 'size', paddingRight: '1.75rem' }}
                value={selection}
                onChange={handleChange}
            >
                {optionElements}
            </Select>
        );
    }
);

interface OptionRowProps {
    value: InputSchemaValue;
    label: string;
    description: string | null | undefined;
    /** Undefined opens the pop-up on hover. */
    isHintOpen: boolean | undefined;
    /** Closes the pop-up the moment the list starts to close. */
    isListOpen: boolean;
    isSelected: boolean;
    isHighlighted: boolean;
    isDisabled: boolean;
    highlightBg: string;
    onPick: (value: InputSchemaValue) => void;
}
const OptionRow = memo(
    ({
        value,
        label,
        description,
        isHintOpen,
        isListOpen,
        isSelected,
        isHighlighted,
        isDisabled,
        highlightBg,
        onPick,
    }: OptionRowProps) => {
        const ref = useRef<HTMLDivElement>(null);
        useEffect(() => {
            if (isHighlighted) {
                ref.current?.scrollIntoView({ block: 'nearest' });
            }
        }, [isHighlighted]);

        return (
            <HintTooltip
                description={description}
                isDisabled={!isListOpen}
                isOpen={isHintOpen}
                placement="right"
            >
                <Box
                    _hover={isDisabled ? undefined : { bg: highlightBg }}
                    aria-disabled={isDisabled}
                    aria-selected={isSelected}
                    bg={isHighlighted ? highlightBg : undefined}
                    borderRadius="md"
                    cursor={isDisabled ? 'not-allowed' : 'pointer'}
                    fontWeight={isSelected ? 'bold' : undefined}
                    opacity={isDisabled ? 0.5 : undefined}
                    px={2}
                    py={0.5}
                    ref={ref}
                    role="option"
                    onClick={isDisabled ? undefined : () => onPick(value)}
                >
                    {label}
                </Box>
            </HintTooltip>
        );
    }
);

interface OptionListProps {
    items: readonly DropDownItem[];
    unavailable: string;
    selection: number;
    /** Undefined for a list without a search field. */
    searchRef: RefObject<HTMLInputElement> | undefined;
    /** Takes the keyboard when there is no search field. */
    listRef: RefObject<HTMLDivElement>;
    isOpen: boolean;
    onPick: (value: InputSchemaValue) => void;
}
const OptionList = memo(
    ({ items, unavailable, selection, searchRef, listRef, isOpen, onPick }: OptionListProps) => {
        const { t } = useTranslation();
        const [query, setQuery] = useState('');
        // the option index of the highlighted option; the first match if it isn't among the matches
        const [highlight, setHighlight] = useState<number | undefined>(selection);
        // after an arrow key, the highlighted option shows its pop-up until the pointer moves
        const [keyboardHint, setKeyboardHint] = useState(false);
        const highlightBg = useColorModeValue('gray.100', 'whiteAlpha.200');

        const filtered = useMemo(() => filterDropDownItems(items, query), [items, query]);
        const selectable = useMemo(
            () =>
                filtered.filter(
                    (item): item is DropDownOptionItem =>
                        item.kind === 'option' && !isUnavailable(unavailable, item.index)
                ),
            [filtered, unavailable]
        );
        const highlightPosition = Math.max(
            0,
            selectable.findIndex((item) => item.index === highlight)
        );
        const highlighted = selectable.at(highlightPosition);

        const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
            if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                event.preventDefault();
                if (selectable.length > 0) {
                    const step = event.key === 'ArrowDown' ? 1 : -1;
                    const next = (highlightPosition + step + selectable.length) % selectable.length;
                    setHighlight(selectable[next].index);
                    setKeyboardHint(true);
                }
            } else if (event.key === 'Enter') {
                event.preventDefault();
                if (highlighted) onPick(highlighted.option.value);
            }
        };

        return (
            <Box
                outline="none"
                ref={listRef}
                tabIndex={-1}
                onKeyDown={onKeyDown}
                onMouseMove={() => setKeyboardHint(false)}
            >
                {searchRef && (
                    <InputGroup size="sm">
                        <InputLeftElement pointerEvents="none">
                            <SearchIcon />
                        </InputLeftElement>
                        <Input
                            borderBottomRadius={0}
                            borderTopRadius="md"
                            placeholder={t('inputs.dropdown.search', 'Search...')}
                            ref={searchRef}
                            spellCheck={false}
                            value={query}
                            variant="filled"
                            onChange={(e) => {
                                setQuery(e.target.value);
                                setHighlight(undefined);
                                setKeyboardHint(false);
                            }}
                        />
                    </InputGroup>
                )}
                <Box
                    fontSize="sm"
                    maxH="calc(min(20rem, 50vh))"
                    overflowY="auto"
                    p={1}
                    role="listbox"
                >
                    {filtered.map((item) => {
                        if (item.kind === 'group') {
                            return item.group.label ? (
                                <Text
                                    fontSize="xs"
                                    fontWeight="bold"
                                    key={`group-${item.index}`}
                                    opacity={0.7}
                                    pt={1}
                                    px={2}
                                >
                                    {item.group.label}
                                </Text>
                            ) : (
                                <Divider
                                    key={`group-${item.index}`}
                                    my={1}
                                />
                            );
                        }
                        const isHighlighted = item === highlighted;
                        return (
                            <OptionRow
                                description={item.option.description}
                                highlightBg={highlightBg}
                                isDisabled={isUnavailable(unavailable, item.index)}
                                isHighlighted={isHighlighted}
                                isHintOpen={keyboardHint ? isHighlighted : undefined}
                                isListOpen={isOpen}
                                isSelected={item.index === selection}
                                key={`option-${item.index}`}
                                label={item.option.option}
                                value={item.option.value}
                                onPick={onPick}
                            />
                        );
                    })}
                    {filtered.length === 0 && (
                        <Text
                            opacity={0.5}
                            px={2}
                            py={0.5}
                        >
                            {t('inputs.dropdown.noMatchingOptions', 'No matching options.')}
                        </Text>
                    )}
                </Box>
            </Box>
        );
    }
);

interface ListDropDownProps extends ListProps {
    isSearchable: boolean;
}
const ListDropDown = memo(
    ({
        items,
        options,
        unavailable,
        selection,
        isDisabled,
        isSearchable,
        onChange,
    }: ListDropDownProps) => {
        const { isOpen, onOpen, onClose } = useDisclosure();
        const searchRef = useRef<HTMLInputElement>(null);
        const listRef = useRef<HTMLDivElement>(null);

        const onPick = useCallback(
            (value: InputSchemaValue) => {
                onClose();
                onChange(value);
            },
            [onChange, onClose]
        );

        return (
            <Popover
                isLazy
                initialFocusRef={isSearchable ? searchRef : listRef}
                isOpen={isOpen}
                placement="bottom-start"
                onClose={onClose}
                onOpen={onOpen}
            >
                <HintTooltip
                    description={options[selection]?.description}
                    isDisabled={isOpen}
                >
                    <Box w="full">
                        <PopoverTrigger>
                            <Button
                                borderColor="inherit"
                                borderRadius="lg"
                                className="nodrag"
                                draggable={false}
                                fontWeight="normal"
                                isDisabled={isDisabled}
                                pl={3}
                                pr={2}
                                rightIcon={<ChevronDownIcon />}
                                size="sm"
                                style={{ contain: 'size' }}
                                variant="outline"
                                w="full"
                            >
                                <Text
                                    as="span"
                                    flex={1}
                                    overflow="hidden"
                                    textAlign="left"
                                    textOverflow="ellipsis"
                                    whiteSpace="nowrap"
                                >
                                    {options[selection]?.option}
                                </Text>
                            </Button>
                        </PopoverTrigger>
                    </Box>
                </HintTooltip>
                <Portal>
                    {/* nokey: the list's keys do not move or delete the node */}
                    <PopoverContent className="nodrag nokey">
                        <OptionList
                            isOpen={isOpen}
                            items={items}
                            listRef={listRef}
                            searchRef={isSearchable ? searchRef : undefined}
                            selection={selection}
                            unavailable={unavailable}
                            onPick={onPick}
                        />
                    </PopoverContent>
                </Portal>
            </Popover>
        );
    }
);

export const DropDown = memo(
    ({ value, onChange, reset, isDisabled, options, groups, testCondition }: DropDownProps) => {
        let selection = options.findIndex((o) => o.value === value);
        if (selection === -1) selection = 0;

        // a string, so it only changes when the set of unavailable options changes
        const unavailable = useMemo(
            () => getUnavailableMask(options, testCondition),
            [options, testCondition]
        );

        useEffect(() => {
            const index = options.findIndex((o) => o.value === value);
            if (index !== -1 && isUnavailable(unavailable, index)) {
                // we can't use reset since the default value might be unavailable too
                // so we search for the first available option
                const firstAvailable = options.find((_, i) => !isUnavailable(unavailable, i));
                if (firstAvailable) {
                    onChange(firstAvailable.value);
                } else {
                    // well, can't do anything then ¯\_(ツ)_/¯
                }
            }
        }, [value, onChange, options, unavailable]);

        // built once per schema, not on every edit of the node's other inputs
        const items = useMemo(() => getDropDownItems(options, groups), [options, groups]);

        const kind = getDropDownKind(options);
        if (kind !== 'select') {
            return (
                <ListDropDown
                    isDisabled={isDisabled}
                    isSearchable={kind === 'search'}
                    items={items}
                    options={options}
                    selection={selection}
                    unavailable={unavailable}
                    onChange={onChange}
                />
            );
        }
        return (
            <SelectDropDown
                isDisabled={isDisabled}
                items={items}
                options={options}
                reset={reset}
                selection={selection}
                unavailable={unavailable}
                onChange={onChange}
            />
        );
    }
);
