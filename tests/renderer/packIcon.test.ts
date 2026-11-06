import { ReactElement } from 'react';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

// No DOM packages are installed: state lives between explicit calls of the component.
const harness = vi.hoisted(() => {
    const state = { value: undefined as unknown };
    return {
        state,
        useState: () => [
            state.value,
            (value: unknown) => {
                state.value = value;
            },
        ],
        useEffect: (effect: () => void) => effect(),
    };
});

vi.mock('react', async (importOriginal) => ({
    ...(await importOriginal<typeof import('react')>()),
    memo: <T>(component: T) => component,
    useEffect: harness.useEffect,
    useState: harness.useState,
}));

type Props = Record<string, unknown>;
type Component = (props: Props) => ReactElement<Props>;

const mount = async (icon: string) => {
    const { IconFactory } = (await import(
        '../../src/renderer/components/CustomIcons'
    )) as unknown as { IconFactory: Component };
    const element = IconFactory({ icon, boxSize: 4 });
    return () => (element.type as Component)(element.props);
};

describe('an icon no schema names', () => {
    beforeEach(() => {
        harness.state.value = undefined;
        vi.resetModules();
    });

    afterEach(() => {
        vi.doUnmock('react-icons/md/index');
        vi.restoreAllMocks();
    });

    test('takes the icon size while its pack loads, then draws from the pack', async () => {
        const drawn = vi.fn();
        vi.doMock('react-icons/md/index', async () => ({
            ...(await vi.importActual<object>('react-icons/md')),
            default: { MdAccessAlarm: drawn },
        }));
        const render = await mount('MdAccessAlarm');

        expect(render().props).toEqual({ boxSize: 4 });

        await vi.waitFor(() => expect(harness.state.value).toBeDefined());
        expect(render().props.as).toBe(drawn);
    });

    test('if the pack fails to load, the failure is logged and it draws as an unknown name', async () => {
        vi.doMock('react-icons/md/index', async () => ({
            ...(await vi.importActual<object>('react-icons/md')),
            get default(): unknown {
                throw new Error('the chunk is missing');
            },
        }));
        const { log } = await import('../../src/common/log');
        const error = vi.spyOn(log, 'error');
        const render = await mount('MdAccessAlarm');
        render();

        await vi.waitFor(() => expect(harness.state.value).toEqual({}));

        expect(error).toHaveBeenCalledOnce();
        expect(render().props.as).toBeUndefined();
    });
});
