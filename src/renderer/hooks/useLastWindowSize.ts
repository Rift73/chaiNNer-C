import { useCallback, useEffect } from 'react';
import { debounce } from '../../common/util';
import { useIpcRendererListener } from './useIpcRendererListener';
import { useMutSetting } from './useSettings';

export const useLastWindowSize = () => {
    const [, setSize] = useMutSetting('lastWindowSize');

    useEffect(() => {
        const listener = debounce(() => {
            setSize((prev) => {
                const { outerWidth: width, outerHeight: height } = window;
                if (prev.maximized || (prev.width === width && prev.height === height)) {
                    return prev;
                }
                return { maximized: false, width, height };
            });
        }, 100);

        window.addEventListener('resize', listener);
        return () => window.removeEventListener('resize', listener);
    }, [setSize]);

    useIpcRendererListener(
        'window-maximized-change',
        useCallback(
            (_, maximized) => {
                setSize((prev) => (prev.maximized === maximized ? prev : { ...prev, maximized }));
            },
            [setSize]
        )
    );
};
