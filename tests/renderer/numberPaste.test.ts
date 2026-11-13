import { NumericLiteralType, StringLiteralType } from '@chainner/navi';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { ColorInput } from '../../src/renderer/components/inputs/ColorInput';
import { NumberInput } from '../../src/renderer/components/inputs/NumberInput';
import { SliderInput } from '../../src/renderer/components/inputs/SliderInput';
import { TextInput } from '../../src/renderer/components/inputs/TextInput';
import { VideoSeekInput } from '../../src/renderer/components/inputs/VideoSeekInput';
import { NumberSetting } from '../../src/renderer/components/settings/components';
import { flush, mount } from './hookRunner';
import type { ReactElement } from 'react';

const stubs = vi.hoisted(() => ({
    menu: undefined as undefined | (() => unknown),
    /** What every context selector reads; none when unset. */
    selected: undefined as unknown,
}));

vi.mock('react', async (importOriginal) => ({
    ...(await importOriginal<typeof import('react')>()),
    ...(await import('./hookRunner')).fakeReact,
}));
vi.mock('@chakra-ui/react', () => ({
    Box: 'box',
    Slider: 'slider-root',
    SliderFilledTrack: 'slider-filled-track',
    SliderThumb: 'slider-thumb',
    SliderTrack: 'slider-track',
    useColorModeValue: <T>(light: T) => light,
    NumberInput: 'number-input',
    NumberInputField: 'number-input-field',
    NumberInputStepper: 'number-input-stepper',
    NumberIncrementStepper: 'number-increment-stepper',
    NumberDecrementStepper: 'number-decrement-stepper',
    Select: 'select',
    Switch: 'switch',
    Popover: 'popover',
    PopoverArrow: 'popover-arrow',
    PopoverBody: 'popover-body',
    PopoverContent: 'popover-content',
    PopoverTrigger: 'popover-trigger',
    Portal: 'portal',
    forwardRef: <T>(component: T) => component,
    Center: 'center',
    Input: 'input',
    Textarea: 'textarea',
    Button: 'button',
    IconButton: 'icon-button',
    Spacer: 'spacer',
    Tooltip: 'tooltip',
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
    useContextSelector: (_: unknown, select: (value: unknown) => unknown) =>
        stubs.selected === undefined ? undefined : select(stubs.selected),
}));
vi.mock('reactflow', () => ({ useReactFlow: () => ({ getNode: () => undefined }) }));
vi.mock('../../src/renderer/contexts/ExecutionContext', () => ({
    ExecutionContext: {},
    ExecutionStatus: { READY: 'ready' },
}));
vi.mock('../../src/renderer/contexts/GlobalNodeState', () => ({
    GlobalContext: {},
    GlobalVolatileContext: {},
}));
vi.mock('../../src/renderer/contexts/FakeExampleContext', () => ({ FakeNodeContext: {} }));
vi.mock('../../src/renderer/hooks/useTypeColor', () => ({ useTypeColor: () => ['#000'] }));
vi.mock('../../src/renderer/appConstants', () => ({ appDataPath: '' }));
vi.mock('../../src/renderer/safeIpc', () => ({ ipcRenderer: {} }));
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
    stubs.selected = undefined;
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

test('the seek bar pastes a frame number, and nothing for text that is none', async () => {
    // the node's last run reported 100 frames at 25 fps
    stubs.selected = {
        isBackendReady: true,
        status: 'ready',
        outputDataMap: new Map([['node', { types: { 12: 100, 4: 25 } }]]),
    };
    const seek = {
        kind: 'video-seek',
        label: 'Frame',
        def: 0,
        framesInput: 1,
        inInput: 2,
        outInput: 3,
        frameCountOutput: 12,
        fpsOutput: 4,
    };

    expect(await paste(VideoSeekInput, seek, ' 12\n')).toEqual([12]);
    expect(await paste(VideoSeekInput, seek, '12.6')).toEqual([13]);
    // past the last frame: the last frame
    expect(await paste(VideoSeekInput, seek, '5000')).toEqual([99]);

    // not numbers
    expect(await paste(VideoSeekInput, seek, '')).toEqual([]);
    expect(await paste(VideoSeekInput, seek, '  \n')).toEqual([]);
    expect(await paste(VideoSeekInput, seek, 'abc')).toEqual([]);
    expect(await paste(VideoSeekInput, seek, 'Infinity')).toEqual([]);
});

test.each([
    ['number', NumberInput, number],
    ['slider', SliderInput, slider],
])('the %s input takes no paste on a locked node or while connected', (_, component, input) => {
    const pasteItem = (state: { isLocked: boolean; isConnected: boolean }) => {
        mount(() =>
            (component as unknown as Component)({
                value: 7,
                setValue: vi.fn(),
                input,
                ...state,
                inputType: new NumericLiteralType(7),
                nodeId: 'node',
            })
        );
        const menu = stubs.menu!() as ReactElement<{
            children: ReactElement<{ isDisabled?: boolean }>[];
        }>;
        return menu.props.children[1].props.isDisabled ?? false;
    };

    expect(pasteItem({ isLocked: false, isConnected: false })).toBe(false);
    expect(pasteItem({ isLocked: true, isConnected: false })).toBe(true);
    expect(pasteItem({ isLocked: false, isConnected: true })).toBe(true);
});

test('the text input takes no paste on a locked node or while connected', () => {
    const text = { kind: 'text', label: 'Text', def: null, minLength: 0, hasHandle: true };
    stubs.selected = { registerInputFlush: () => () => {}, zoom: 1 };
    const pasteItem = (state: { isLocked: boolean; isConnected: boolean }) => {
        mount(() =>
            (TextInput as unknown as Component)({
                value: 'a',
                setValue: vi.fn(),
                resetValue: vi.fn(),
                input: text,
                ...state,
                inputType: new StringLiteralType('a'),
                size: { width: 1, height: 1 },
                setSize: vi.fn(),
                nodeId: 'node',
            })
        );
        const menu = stubs.menu!() as ReactElement<{
            children: ReactElement<{ isDisabled?: boolean }>[];
        }>;
        return menu.props.children[1].props.isDisabled ?? false;
    };

    expect(pasteItem({ isLocked: false, isConnected: false })).toBe(false);
    expect(pasteItem({ isLocked: true, isConnected: false })).toBe(true);
    expect(pasteItem({ isLocked: false, isConnected: true })).toBe(true);
});

test('the color input neither opens nor applies a color on a locked node', () => {
    const red = JSON.stringify({ kind: 'rgb', values: [1, 0, 0] });
    const blue = { kind: 'rgb', values: [0, 0, 1] };
    const picker = (isLocked: boolean) => {
        const setValue = vi.fn();
        const { result } = mount(() =>
            (ColorInput as unknown as Component)({
                value: red,
                setValue,
                input: { kind: 'color', label: 'Color', def: red, channels: null },
                isLocked,
                isConnected: false,
                definitionType: new StringLiteralType(red),
            })
        );
        type Element = ReactElement<Record<string, unknown>>;
        // the color picker's popover is the element that takes the color kinds
        const find = (element: unknown): Element | undefined => {
            if (!element || typeof element !== 'object') return undefined;
            const el = element as Element;
            if ('kinds' in el.props) return el;
            const { children } = el.props as { children?: unknown };
            for (const child of Array.isArray(children) ? children : [children]) {
                const found = find(child);
                if (found) return found;
            }
            return undefined;
        };
        const popoverElement = find(result)!;
        const popover = (popoverElement.type as Component)(popoverElement.props) as Element;
        (popoverElement.props.onChange as (c: unknown) => void)(blue);
        return { isOpen: popover.props.isOpen, applied: setValue.mock.calls.length > 0 };
    };

    expect(picker(false)).toEqual({ isOpen: undefined, applied: true });
    expect(picker(true)).toEqual({ isOpen: false, applied: false });
});

test('a number setting takes only finite numbers, as the number inputs do', () => {
    const typed = (text: string): unknown[] => {
        const setValue = vi.fn();
        const container = (NumberSetting as unknown as Component)({
            setting: { label: 'Threads', description: '', min: 0, max: 64 },
            value: 4,
            setValue,
        }) as ReactElement<{ children: ReactElement<{ onChange: (text: string) => void }> }>;
        container.props.children.props.onChange(text);
        return setValue.mock.calls.map((call: unknown[]) => call[0]);
    };

    expect(typed('8')).toEqual([8]);
    expect(typed('abc')).toEqual([]);
    expect(typed('1e999')).toEqual([]);
    expect(typed('-1e999')).toEqual([]);
});

test('double-clicking a disabled slider keeps its value', async () => {
    const { StyledSlider } = await vi.importActual<
        typeof import('../../src/renderer/components/inputs/elements/StyledSlider')
    >('../../src/renderer/components/inputs/elements/StyledSlider');
    const { LINEAR_SCALE } = await import('../../src/renderer/helpers/sliderScale');
    const doubleClick = (isDisabled: boolean): unknown[] => {
        const onChangeEnd = vi.fn();
        const { result } = mount(() =>
            (StyledSlider as unknown as Component)({
                style: { type: 'no-fill' },
                scale: LINEAR_SCALE,
                isDisabled,
                min: 0,
                max: 10,
                def: 5,
                value: 2,
                step: 1,
                onChange: vi.fn(),
                onChangeEnd,
            })
        );
        (result as ReactElement<{ onDoubleClick?: () => void }>).props.onDoubleClick?.();
        return onChangeEnd.mock.calls.map((call: unknown[]) => call[0]);
    };

    expect(doubleClick(false)).toEqual([5]);
    expect(doubleClick(true)).toEqual([]);
});
