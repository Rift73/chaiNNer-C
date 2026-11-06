import { evaluate } from 'mathjs';
import { ReactElement } from 'react';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import type { NumberInputProps } from '@chakra-ui/react';

// No DOM packages are installed: refs live between explicit calls of the component.
const harness = vi.hoisted(() => {
    const state = { slots: [] as { current: unknown }[], cursor: 0 };
    return {
        state,
        useRef: <T>(initial: T): { current: T } => {
            const index = state.cursor;
            state.cursor += 1;
            if (index === state.slots.length) state.slots.push({ current: initial });
            return state.slots[index] as { current: T };
        },
    };
});

vi.mock('react', async (importOriginal) => ({
    ...(await importOriginal<typeof import('react')>()),
    memo: <T>(component: T) => component,
    useRef: harness.useRef,
    useCallback: <T>(callback: T) => callback,
    useEffect: () => {},
}));
vi.mock('use-context-selector', () => ({ useContextSelector: () => () => () => {} }));
vi.mock('../../src/renderer/contexts/GlobalNodeState', () => ({ GlobalContext: {} }));
vi.mock('@chakra-ui/react', () => ({
    InputGroup: 'input-group',
    InputLeftAddon: 'input-left-addon',
    NumberInput: 'number-input',
    NumberInputField: 'number-input-field',
    NumberInputStepper: 'number-input-stepper',
    NumberIncrementStepper: 'number-increment-stepper',
    NumberDecrementStepper: 'number-decrement-stepper',
}));

type KeyHandler = (e: { key: string }) => void;

const mount = async (text: string, limits = { min: -1e9, max: 1e9 }) => {
    const { AdvancedNumberInput } = await import(
        '../../src/renderer/components/inputs/elements/AdvanceNumberInput'
    );
    const setInput = vi.fn<[number], void>();
    let inputString = text;
    const render = () => {
        harness.state.cursor = 0;
        const tree = AdvancedNumberInput({
            ...limits,
            precision: 6,
            controlsStep: 1,
            defaultValue: 1,
            hideTrailingZeros: true,
            inputString,
            setInput,
            setInputString: (value: string) => {
                inputString = value;
            },
        }) as ReactElement<{ children: ReactElement<NumberInputProps>[] }>;
        return tree.props.children[1];
    };
    const pressEnter = () => {
        const field = (render().props.children as ReactElement<{ onKeyDown: KeyHandler }>[])[0];
        field.props.onKeyDown({ key: 'Enter' });
    };
    const retype = (value: string) => {
        inputString = value;
        render();
    };
    return { focus: () => render().props.onFocus, pressEnter, retype, setInput };
};

describe('number input expressions', () => {
    beforeEach(() => {
        harness.state.slots = [];
        vi.resetModules();
    });

    afterEach(() => {
        vi.doUnmock('mathjs');
        vi.restoreAllMocks();
    });

    test.each(['12', '1.', '.5', '+5', '-5', '007', '1e3', '1.5E-2', ' 7 ', '0.1'])(
        'the plain number %j commits at once, with the value mathjs reads',
        async (text) => {
            const { pressEnter, setInput } = await mount(text);

            pressEnter();

            expect(setInput).toHaveBeenCalledOnce();
            expect(setInput).toHaveBeenCalledWith(evaluate(text));
        }
    );

    test('an expression commits once mathjs has loaded', async () => {
        const { pressEnter, setInput } = await mount('2*3 + 2**3');

        pressEnter();
        expect(setInput).not.toHaveBeenCalled();

        await vi.waitFor(() => expect(setInput).toHaveBeenCalledWith(14), { timeout: 20000 });
        expect(setInput).toHaveBeenCalledOnce();
    });

    test.each(['1e999', '-1e999', '1/0', '-1/0'])(
        'without limits, the infinite number %j is an invalid one: a chain cannot hold it',
        async (text) => {
            const { pressEnter, setInput } = await mount(text, { min: -Infinity, max: Infinity });

            pressEnter();

            await new Promise((resolve) => {
                setTimeout(resolve, 500);
            });
            await import('mathjs');
            await new Promise((resolve) => {
                setTimeout(resolve, 0);
            });
            expect(setInput).not.toHaveBeenCalled();
        }
    );

    test('mathjs starts loading when an input gets focus, not before', async () => {
        const actual = await vi.importActual<typeof import('mathjs')>('mathjs');
        const loaded = vi.fn();
        vi.doMock('mathjs', () => {
            loaded();
            return actual;
        });
        const { focus } = await mount('2*3');
        expect(loaded).not.toHaveBeenCalled();

        focus()?.({} as never);

        await vi.waitFor(() => expect(loaded).toHaveBeenCalledOnce());
    });

    test('text changed while mathjs loads is not committed, the new text is', async () => {
        const { pressEnter, retype, setInput } = await mount('2*3');

        pressEnter();
        retype('2*4');
        pressEnter();

        await vi.waitFor(() => expect(setInput).toHaveBeenCalledWith(8), { timeout: 20000 });
        expect(setInput).toHaveBeenCalledOnce();
    });

    test('without mathjs an expression reads as an invalid one, and the failure is logged', async () => {
        vi.doMock('mathjs', () => {
            throw new Error('the chunk is missing');
        });
        const { log } = await import('../../src/common/log');
        const error = vi.spyOn(log, 'error');
        const { pressEnter, setInput } = await mount('2*3');

        pressEnter();

        await vi.waitFor(() => expect(setInput).toHaveBeenCalledWith(2));
        expect(error).toHaveBeenCalledOnce();
    });
});
