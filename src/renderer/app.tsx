import { Box, Center, ChakraProvider, ColorModeScript, Spinner } from '@chakra-ui/react';
import TimeAgo from 'javascript-time-ago';
import en from 'javascript-time-ago/locale/en.json';
import { memo, useState } from 'react';
import { BackendData } from '../common/Backend';
import { PythonInfo } from '../common/common-types';
import { log } from '../common/log';
import { ChainnerSettings } from '../common/settings/settings';
import { AlertBoxProvider } from './contexts/AlertBoxContext';
import { BackendProvider } from './contexts/BackendContext';
import { ContextMenuProvider } from './contexts/ContextMenuContext';
import { HotkeysProvider } from './contexts/HotKeyContext';
import { SettingsProvider } from './contexts/SettingsContext';
import { useAsyncEffect } from './hooks/useAsyncEffect';
import { Main } from './main';
import { ipcRenderer } from './safeIpc';
import { darktheme } from './theme';
import './i18n';

TimeAgo.addLocale(en);
TimeAgo.setDefaultLocale(en.locale);

const LoadingComponent = memo(() => (
    <Box
        h="full"
        w="full"
    >
        <Center
            h="full"
            w="full"
        >
            <Spinner />
        </Center>
    </Box>
));

export const App = memo(() => {
    const [url, setUrl] = useState<string>();
    useAsyncEffect(
        () => ({
            supplier: () => ipcRenderer.invoke('get-backend-url'),
            successEffect: setUrl,
        }),
        []
    );

    const [settings, setSettings] = useState<ChainnerSettings>();
    useAsyncEffect(
        () => ({
            supplier: () => ipcRenderer.invoke('get-settings'),
            successEffect: setSettings,
        }),
        []
    );

    const [pythonInfo, setPythonInfo] = useState<PythonInfo>();
    useAsyncEffect(
        () => ({
            supplier: () => ipcRenderer.invoke('get-python'),
            successEffect: setPythonInfo,
        }),
        []
    );

    // The node list of the last start lets the editor show at once, without waiting for the
    // backend; `null` when there is none.
    const [cachedData, setCachedData] = useState<BackendData | null>();
    useAsyncEffect(
        () => ({
            supplier: () => ipcRenderer.invoke('get-ui-cache'),
            successEffect: (data) => setCachedData(data ?? null),
            catchEffect: (error) => {
                log.error('Unable to read the node list cache.', error);
                setCachedData(null);
            },
        }),
        []
    );

    return (
        <ChakraProvider theme={darktheme}>
            <ColorModeScript initialColorMode={darktheme.config.initialColorMode} />
            <HotkeysProvider>
                <ContextMenuProvider>
                    <AlertBoxProvider>
                        {!url || !settings || !pythonInfo || cachedData === undefined ? (
                            <LoadingComponent />
                        ) : (
                            <SettingsProvider initialSettings={settings}>
                                <BackendProvider
                                    cachedData={cachedData ?? undefined}
                                    pythonInfo={pythonInfo}
                                    url={url}
                                >
                                    <Main />
                                </BackendProvider>
                            </SettingsProvider>
                        )}
                    </AlertBoxProvider>
                </ContextMenuProvider>
            </HotkeysProvider>
        </ChakraProvider>
    );
});
