import { NumericLiteralType } from '@chainner/navi';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { NumberInput } from '../../src/renderer/components/inputs/NumberInput';
import { SliderInput } from '../../src/renderer/components/inputs/SliderInput';
import { flush, mount } from './hookRunner';
import type { ReactElement } from 'react';

const stubs = vi.hoisted(() => ({
    menu: undefined as undefined | (() => unknown),
}));

vi.mock('react', async (importOriginal) => ({
    ...(await importOriginal<typeof import('react')>()),
    ...(await import('./hookRunner')).fakeReact,
}));
vi.mock('@chakra-ui/react', () => ({
    HStack: 'h-stack',
    VStack: 'v-stack',
    MenuItem: 'menu-item',
    MenuList: 'menu-list',
    Text: 'text',
}));
vi.mock('react-i18next', () => ({
    useTranslation: () => ({ t: (_key: string, text: string) => text }),
}));
vi.mock('use-context-selector', () => ({
    createContext: () => ({}),
    useContext: () => ({ conditionallyInactive: false }),
    useContextSelector: () => undefined,
}));
vi.mock('../../src/renderer/contexts/BackendContext', () => ({ BackendContext: {} }));
vi.mock('../../src/renderer/contexts/InputContext', () => ({ InputContext: {} }));
vi.mock('../../src/renderer/hooks/useContextMenu', () => ({
    useContextMenu: (render: () => unknown) => {
        stubs.menu = render;
        return {};
    },
}));
vi.mock('../../src/renderer/hooks/useInputRefactor', () => ({ useInputRefactor: () => null }));
vi.mock('../../src/renderer/components/inputs/elements/AdvanceNumberInput', () => ({
    AdvancedNumberInput: 'number-field',
}));
vi.mock('../../src/renderer/components/inputs/elements/StyledSlider', () => ({
    StyledSlider: 'slider',
}));
vi.mock('../../src/renderer/components/inputs/InputContainer', () => ({
    AutoLabel: 'auto-label',
    WithLabel: 'with-label',
    WithoutLabel: 'without-label',
}));

const number = {
    kind: 'number',
    label: 'Value',
    def: 1,
    min: null as number | null,
    max: null as number | null,
    precision: 2,
    controlsStep: 1,
    unit: null,
    hideTrailingZeros: true,
};
const slider = {
    ...number,
    kind: 'slider',
    min: -100,
    max: 100,
    sliderStep: 1,
    scale: 'linear',
    ends: [null, null],
    noteExpression: null,
    gradient: null,
};

type Component = (props: Record<string, unknown>) => unknown;
const paste = async (component: unknown, input: unknown, text: string): Promise<unknown[]> => {
    const setValue = vi.fn();
    mount(() =>
        (component as Component)({
            value: 7,
            setValue,
            input,
            isConnected: false,
            isLocked: false,
            inputType: new NumericLiteralType(7),
            nodeId: 'node',
        })
    );
    vi.stubGlobal('navigator', { clipboard: { readText: () => Promise.resolve(text) } });
    const menu = stubs.menu!() as ReactElement<{
        children: ReactElement<{ onClick: () => void }>[];
    }>;
    menu.props.children[1].props.onClick();
    await Promise.resolve();
    await Promise.resolve();
    flush();
    return setValue.mock.calls.map((call: unknown[]) => call[0]);
};

beforeEach(() => {
    stubs.menu = undefined;
});

afterEach(() => {
    vi.unstubAllGlobals();
});

test.each([
    ['number', NumberInput, number],
    ['slider', SliderInput, slider],
])('the %s input pastes the numbers it takes, and only those', async (_, component, input) => {
    expect(await paste(component, input, '12.5')).toEqual([12.5]);
    expect(await paste(component, input, ' 12.5\n')).toEqual([12.5]);

    // not numbers
    expect(await paste(component, input, '')).toEqual([]);
    expect(await paste(component, input, '  \n')).toEqual([]);
    expect(await paste(component, input, 'abc')).toEqual([]);

    // in the input's precision, as when typed
    expect(await paste(component, input, '1.23456')).toEqual([1.23]);
    expect(await paste(component, { ...input, precision: 0 }, '12.5')).toEqual([13]);
});

test('an input without limits takes no infinite number from the clipboard', async () => {
    expect(await paste(NumberInput, number, 'Infinity')).toEqual([]);
    expect(await paste(NumberInput, number, '-Infinity')).toEqual([]);
    expect(await paste(NumberInput, number, '1e999')).toEqual([]);
});

test('a limit still refuses a pasted number', async () => {
    expect(await paste(SliderInput, slider, '1000')).toEqual([]);
    expect(await paste(NumberInput, { ...number, min: 0 }, '-1')).toEqual([]);
});
