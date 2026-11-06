import {
    InputGroup,
    InputLeftAddon,
    NumberDecrementStepper,
    NumberIncrementStepper,
    NumberInput,
    NumberInputField,
    NumberInputStepper,
} from '@chakra-ui/react';
import { CSSProperties, MouseEventHandler, memo, useCallback, useEffect, useRef } from 'react';
import { useContextSelector } from 'use-context-selector';
import { log } from '../../../../common/log';
import { lazy, noop } from '../../../../common/util';
import { GlobalContext } from '../../../contexts/GlobalNodeState';
import './AdvancedNumberInput.scss';

const validChars = /^[\w.+\-*()^!%&|~ /]$/iu;
const isValidChar = (c: string): boolean => validChars.test(c);

// Text mathjs would read as this same number. Anything else is an expression, which needs mathjs.
const plainNumber = /^\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?\s*$/iu;

let evaluateExpression: ((s: string) => number) | undefined;

// mathjs takes long to start and only expressions need it, so it loads when an input gets focus,
// not with the app. Without it, an expression reads as an invalid one does.
const loadMathjs = lazy(async () => {
    try {
        const { evaluate, isBigNumber, isComplex, isFraction, isNumber, isUnit, number } =
            await import('mathjs');
        evaluateExpression = (s) => {
            try {
                const result = evaluate(s.replace(/\*\*/g, ' ^ ')) as unknown;
                if (isNumber(result)) return result;
                // ignore units
                if (isUnit(result)) return result.toNumber();
                // just return the real part of complex numbers
                if (isComplex(result)) return result.re;
                // just return the real part of complex numbers
                if (isBigNumber(result)) return result.toNumber();
                if (isFraction(result)) return number(result);
            } catch {
                // noop
            }
            return parseFloat(s);
        };
    } catch (error) {
        log.error('Unable to load mathjs. Number inputs only read plain numbers.', error);
        evaluateExpression = parseFloat;
    }
});

const startLoadingMathjs = () => {
    loadMathjs().catch(log.error);
};

/** The number the text stands for, or undefined while its expression waits for mathjs. */
const parseNumberString = (s: string): number | undefined => {
    return plainNumber.test(s) ? parseFloat(s) : evaluateExpression?.(s);
};

const clamp = (value: number, min?: number | null, max?: number | null): number => {
    if (min != null && value < min) return min;
    if (max != null && value > max) return max;
    return value;
};

interface AdvancedNumberInputProps {
    unit?: string | null;
    max: number;
    min: number;
    precision: number;
    controlsStep: number;
    hideTrailingZeros: boolean;

    defaultValue: number;
    isDisabled?: boolean;
    small?: true;

    inputString: string;
    setInputString: (input: string) => void;
    setInput: (value: number) => void;
    /** If given, committing an empty input calls this instead of using the default value. */
    onClear?: () => void;

    onContextMenu?: MouseEventHandler<HTMLElement> | undefined;
    inputWidth?: string;
    inputHeight?: string;
    noRepeatOnBlur?: boolean;

    alignSelf?: CSSProperties['alignSelf'];
}

export const AdvancedNumberInput = memo(
    ({
        unit,
        max,
        min,
        precision,
        controlsStep,
        hideTrailingZeros,

        defaultValue,
        isDisabled,
        small,

        inputString,
        setInputString,
        setInput,
        onClear,

        onContextMenu,
        inputWidth,
        inputHeight,
        noRepeatOnBlur = false,

        alignSelf,
    }: AdvancedNumberInputProps) => {
        const changedSinceCommit = useRef(false);
        const commitLoaded = useRef<(waiting: string) => void>(noop);
        const isCleared = onClear !== undefined && !inputString.trim();
        const clear = (): void => {
            changedSinceCommit.current = false;
            onClear?.();
        };
        const getNumericValue = (): number | undefined => {
            if (isCleared) return undefined;
            const rawNumber = inputString.trim() ? parseNumberString(inputString) : defaultValue;
            if (rawNumber === undefined) {
                const waiting = inputString;
                loadMathjs()
                    .then(() => commitLoaded.current(waiting))
                    .catch(log.error);
                return undefined;
            }
            const valAsNumber = precision > 0 ? rawNumber : Math.round(rawNumber);

            const clamped = clamp(valAsNumber, min, max);
            // without limits, an infinite number stays: a chain cannot hold one, it saves as null
            if (Number.isFinite(clamped)) {
                return Number(clamped.toFixed(precision));
            }
        };
        const setValue = (value: number): void => {
            // eslint-disable-next-line @typescript-eslint/no-use-before-define
            cancelAutoCommit();

            changedSinceCommit.current = false;
            setInput(value);
            const formatted = hideTrailingZeros ? String(value) : value.toFixed(precision);
            setInputString(formatted);
        };
        commitLoaded.current = (waiting) => {
            // the text may have changed while mathjs loaded
            if (waiting !== inputString) return;
            const value = getNumericValue();
            if (value !== undefined) {
                setValue(value);
            }
        };

        // auto commit system to set value after a delay
        const timerRef = useRef<NodeJS.Timeout | null>(null);
        const autoCommit = () => {
            if (timerRef.current !== null) {
                clearTimeout(timerRef.current);
                timerRef.current = null;
            }

            timerRef.current = setTimeout(() => {
                const value = getNumericValue();
                if (value !== undefined) {
                    setValue(value);
                }
            }, 500);
        };
        const cancelAutoCommit = useCallback(() => {
            if (timerRef.current !== null) {
                clearTimeout(timerRef.current);
                timerRef.current = null;
            }
        }, []);
        useEffect(() => {
            cancelAutoCommit();
        }, [inputString, cancelAutoCommit]);

        const registerInputFlush = useContextSelector(GlobalContext, (c) => c.registerInputFlush);
        const flushDraft = useRef(() => {});
        flushDraft.current = () => {
            if (!changedSinceCommit.current || isDisabled || noRepeatOnBlur) return;
            cancelAutoCommit();
            if (isCleared) {
                clear();
                return;
            }
            const value = getNumericValue();
            if (value !== undefined) {
                setValue(value);
            }
        };
        useEffect(() => registerInputFlush(() => flushDraft.current()), [registerInputFlush]);

        // event handlers
        const onChange = (value: string) => {
            changedSinceCommit.current = true;
            setInputString(value);
        };
        const onBlur = () => {
            cancelAutoCommit();
            if (noRepeatOnBlur) return;
            if (isCleared) {
                clear();
                return;
            }
            const value = getNumericValue();
            if (value !== undefined) {
                // Make sure the input value has been altered so onChange gets correct value if adjustment needed
                setTimeout(() => setValue(value), 0);
            }
        };
        const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
            cancelAutoCommit();
            if (e.key === 'Enter') {
                if (isCleared) {
                    clear();
                    return;
                }
                const value = getNumericValue();
                if (value !== undefined) {
                    setValue(value);
                }
            }
        };
        const onClick = (e: React.MouseEvent<HTMLElement>) => {
            if (e.target instanceof HTMLElement || e.target instanceof SVGElement) {
                const buttonSelector = 'div.small-stepper > div[role=button]';
                const isStepperButton = e.target.matches(`${buttonSelector}, ${buttonSelector} *`);
                if (isStepperButton) {
                    autoCommit();
                }
            }
        };

        if (small) {
            return (
                <InputGroup
                    alignSelf={alignSelf}
                    mx={0}
                    size="xs"
                    w="fit-content"
                    onClick={onClick}
                    onContextMenu={onContextMenu}
                    onMouseDown={cancelAutoCommit}
                >
                    {unit && (
                        <InputLeftAddon
                            borderLeftRadius="md"
                            h={inputHeight}
                            px={1}
                            w="fit-content"
                        >
                            {unit}
                        </InputLeftAddon>
                    )}
                    <NumberInput
                        borderRadius="md"
                        className="nodrag"
                        defaultValue={defaultValue}
                        draggable={false}
                        isDisabled={isDisabled}
                        isValidCharacter={isValidChar}
                        max={max}
                        min={min}
                        size="xs"
                        step={controlsStep}
                        value={inputString}
                        onBlur={onBlur}
                        onChange={onChange}
                        onFocus={startLoadingMathjs}
                    >
                        <NumberInputField
                            borderLeftRadius={unit ? 0 : 'md'}
                            borderRightRadius="md"
                            h={inputHeight}
                            m={0}
                            pl={1}
                            pr={4}
                            py={1}
                            size={1}
                            w={inputWidth}
                            onKeyDown={onKeyDown}
                        />
                        <NumberInputStepper
                            className="small-stepper"
                            w={4}
                        >
                            <NumberIncrementStepper />
                            <NumberDecrementStepper />
                        </NumberInputStepper>
                    </NumberInput>
                </InputGroup>
            );
        }

        return (
            <InputGroup
                alignSelf={alignSelf}
                size="sm"
                w="full"
                onClick={onClick}
                onContextMenu={onContextMenu}
                onMouseDown={cancelAutoCommit}
            >
                {unit && (
                    <InputLeftAddon
                        borderLeftRadius="lg"
                        px={2}
                        w="fit-content"
                    >
                        {unit}
                    </InputLeftAddon>
                )}
                <NumberInput
                    borderRadius="lg"
                    className="nodrag"
                    defaultValue={defaultValue}
                    draggable={false}
                    isDisabled={isDisabled}
                    isValidCharacter={isValidChar}
                    max={max}
                    min={min}
                    size="sm"
                    step={controlsStep}
                    value={inputString}
                    w="full"
                    onBlur={onBlur}
                    onChange={onChange}
                    onFocus={startLoadingMathjs}
                >
                    <NumberInputField
                        borderLeftRadius={unit ? 0 : 'lg'}
                        borderRightRadius="lg"
                        pl={unit ? 2 : undefined}
                        pr="var(--number-input-stepper-width)"
                        size={1}
                        w={inputWidth}
                        onKeyDown={onKeyDown}
                    />
                    <NumberInputStepper className="small-stepper">
                        <NumberIncrementStepper />
                        <NumberDecrementStepper />
                    </NumberInputStepper>
                </NumberInput>
            </InputGroup>
        );
    }
);
