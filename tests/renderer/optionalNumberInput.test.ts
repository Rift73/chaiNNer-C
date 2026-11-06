import { NeverType } from '@chainner/navi';
import { ReactElement } from 'react';
import { beforeEach, expect, test, vi } from 'vitest';
import { AdvancedNumberInput } from '../../src/renderer/components/inputs/elements/AdvanceNumberInput';
import { NumberInput } from '../../src/renderer/components/inputs/NumberInput';
import type { NumberInputProps } from '@chakra-ui/react';

// No DOM packages are installed: hook state lives between explicit calls of the component, and
// each call runs its effects.
const harness = vi.hoisted(() => {
    const state = {
        slots: [] as { current: unknown }[],
        cursor: 0,
        flush: () => {},
    };
    const useRef = <T>(initial: T): { current: T } => {
        const index = state.cursor;
        state.cursor += 1;
        if (index === state.slots.length) state.slots.push({ current: initial });
        return state.slots[index] as { current: T };
    };
    const useState = <T>(initial: T) => {
        const cell = useRef(initial);
        return [
            cell.current,
            (update: T | ((old: T) => T)) => {
                cell.current =
                    typeof update === 'function' ? (update as (old: T) => T)(cell.current) : update;
            },
        ] as const;
    };
    return { state, useRef, useState };
});

vi.mock('react', async (importOriginal) => ({
    ...(await importOriginal<typeof import('react')>()),
    memo: <T>(component: T) => component,
    useRef: harness.useRef,
    useState: harness.useState,
    useCallback: <T>(callback: T) => callback,
    useEffect: (effect: () => void) => effect(),
}));
vi.mock('use-context-selector', () => ({
    useContextSelector: () => (flush: () => void) => {
        harness.state.flush = flush;
        return () => {};
    },
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: () => '' }) }));
vi.mock('../../src/renderer/contexts/GlobalNodeState', () => ({ GlobalContext: {} }));
vi.mock('../../src/renderer/hooks/useContextMenu', () => ({
    useContextMenu: () => ({ onContextMenu: () => {} }),
}));
vi.mock('../../src/renderer/hooks/useInputRefactor', () => ({ useInputRefactor: () => null }));
vi.mock('../../src/renderer/components/inputs/InputContainer', () => ({
    AutoLabel: 'auto-label',
}));
vi.mock('@chakra-ui/react', () => ({
    HStack: 'h-stack',
    InputGroup: 'input-group',
    InputLeftAddon: 'input-left-addon',
    MenuItem: 'menu-item',
    MenuList: 'menu-list',
    NumberInput: 'number-input',
    NumberInputField: 'number-input-field',
    NumberInputStepper: 'number-input-stepper',
    NumberIncrementStepper: 'number-increment-stepper',
    NumberDecrementStepper: 'number-decrement-stepper',
}));

type KeyHandler = (e: { key: string }) => void;

const mountField = (text: string, optional: boolean) => {
    const setInput = vi.fn<[number], void>();
    const onClear = vi.fn<[], void>();
    let inputString = text;
    const render = () => {
        harness.state.cursor = 0;
        const tree = AdvancedNumberInput({
            min: -100,
            max: 100,
            precision: 2,
            controlsStep: 1,
            defaultValue: 7,
            hideTrailingZeros: true,
            inputString,
            setInput,
            onClear: optional ? onClear : undefined,
            setInputString: (value: string) => {
                inputString = value;
            },
        }) as ReactElement<{ children: ReactElement<NumberInputProps>[] }>;
        return tree.props.children[1];
    };
    return {
        setInput,
        onClear,
        render,
        pressEnter: () => {
            const field = (render().props.children as ReactElement<{ onKeyDown: KeyHandler }>[])[0];
            field.props.onKeyDown({ key: 'Enter' });
        },
        blur: () => render().props.onBlur?.({} as never),
    };
};

interface FieldProps {
    readonly inputString: string;
    readonly defaultValue: number;
    readonly onClear?: () => void;
}

const mountWrapper = (def: number | null, optional: boolean) => {
    const setValue = vi.fn<[number], void>();
    const resetValue = vi.fn<[], void>();
    const render = (value: number | undefined) => {
        harness.state.cursor = 0;
        const tree = NumberInput({
            value,
            setValue,
            resetValue,
            input: {
                kind: 'number',
                label: 'a',
                optional,
                hasHandle: true,
                def,
                precision: 100,
                controlsStep: 1,
                hideTrailingZeros: true,
            },
            isConnected: false,
            isLocked: false,
            inputType: NeverType.instance,
            nodeId: 'node',
        } as never) as ReactElement<{
            children: ReactElement<{ children: ReactElement<FieldProps> }>;
        }>;
        return tree.props.children.props.children.props;
    };
    return { setValue, resetValue, render };
};

beforeEach(() => {
    harness.state.slots = [];
    harness.state.flush = () => {};
});

test.each(['', '   '])('committing the empty text %j clears an optional field', (text) => {
    for (const commit of ['pressEnter', 'blur'] as const) {
        const field = mountField(text, true);
        field[commit]();
        expect(field.onClear, commit).toHaveBeenCalledOnce();
        expect(field.setInput, commit).not.toHaveBeenCalled();
    }
});

test('an optional field with a number commits the number', () => {
    const field = mountField('2.5', true);
    field.pressEnter();
    expect(field.setInput).toHaveBeenCalledOnce();
    expect(field.setInput).toHaveBeenCalledWith(2.5);
    expect(field.onClear).not.toHaveBeenCalled();
});

test('Save clears a field emptied a moment before, once', () => {
    const field = mountField('2.5', true);
    field.render().props.onChange?.('', 0);
    field.render();
    harness.state.flush();
    expect(field.onClear).toHaveBeenCalledOnce();
    expect(field.setInput).not.toHaveBeenCalled();
    field.render();
    harness.state.flush();
    expect(field.onClear).toHaveBeenCalledOnce();
});

test('Save leaves an optional field that was not edited', () => {
    const field = mountField('', true);
    field.render();
    harness.state.flush();
    expect(field.onClear).not.toHaveBeenCalled();
    expect(field.setInput).not.toHaveBeenCalled();
});

test.each(['', '   '])('a required field reads %j as its default, as it always did', (text) => {
    const field = mountField(text, false);
    field.pressEnter();
    expect(field.setInput).toHaveBeenCalledOnce();
    expect(field.setInput).toHaveBeenCalledWith(7);
    expect(field.onClear).not.toHaveBeenCalled();
});

test('an optional input with no value shows an empty field and sets no value', () => {
    const { render, setValue, resetValue } = mountWrapper(null, true);
    const field = render(undefined);
    expect(field.inputString).toBe('');
    expect(field.defaultValue).toBe(0);
    expect(field.onClear).toBe(resetValue);
    expect(setValue).not.toHaveBeenCalled();
    render(undefined);
    expect(setValue).not.toHaveBeenCalled();
});

test('an optional input shows its value, zero included, and empties when the value goes', () => {
    const { render, setValue } = mountWrapper(null, true);
    expect(render(0).inputString).toBe('0');
    render(undefined);
    expect(render(undefined).inputString).toBe('');
    expect(setValue).not.toHaveBeenCalled();
});

test('a required input takes its default, shows it and cannot be cleared', () => {
    const { render, setValue } = mountWrapper(5, false);
    const field = render(undefined);
    expect(field.inputString).toBe('5');
    expect(field.defaultValue).toBe(5);
    expect(field.onClear).toBeUndefined();
    expect(setValue).toHaveBeenCalledOnce();
    expect(setValue).toHaveBeenCalledWith(5);
});

test('a required input keeps a value of zero', () => {
    const { render, setValue } = mountWrapper(5, false);
    expect(render(0).inputString).toBe('0');
    expect(setValue).not.toHaveBeenCalled();
});
