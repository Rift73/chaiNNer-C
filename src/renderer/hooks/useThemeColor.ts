import { useColorMode } from '@chakra-ui/react';
import { useMemo } from 'react';
import { lazy } from '../../common/util';
import { useSettings } from './useSettings';

// A live declaration: it reads the current values of the theme's variables.
const rootStyle = lazy(() => getComputedStyle(document.documentElement));

export const useThemeColor = (name: `--${string}`): string => {
    const { colorMode } = useColorMode();
    const { theme } = useSettings();
    return useMemo(() => {
        return rootStyle().getPropertyValue(name).trim();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [colorMode, name, theme]);
};
