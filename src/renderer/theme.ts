import { Theme, extendTheme } from '@chakra-ui/react';

// This is the initial theme config on startup
const dark = {
    initialColorMode: 'dark',
    useSystemColorMode: false,
} as const;

const grays = [50, 100, 200, 300, 400, 500, 600, 650, 700, 750, 800, 850, 900];
const colors = {
    gray: Object.fromEntries(grays.map((v) => [v, `var(--theme-${v})`])),
};

const fonts = {
    heading: `Open Sans, sans-serif`,
    body: `Open Sans, sans-serif`,
    monospace: `Roboto-Mono, monospace`,
};

export const darktheme = extendTheme({ config: dark, colors, fonts } as const) as Theme;
