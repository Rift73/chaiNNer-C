import { Box, Center, ChakraProvider, ColorModeScript, Spinner } from '@chakra-ui/react';
import TimeAgo from 'javascript-time-ago';
import en from 'javascript-time-ago/locale/en.json';
import { memo, useLayoutEffect, useState } from 'react';
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
import { useSettings } from './hooks/useSettings';
import { applyLanguage } from './i18n';
import { Main } from './main';
import { ipcRenderer } from './safeIpc';
import { darktheme } from './theme';

TimeAgo.addLocale(en);
TimeAgo.setDefaultLocale(en.locale);

// Applies a change of the language setting. Components using useTranslation re-render with the new
// language.
const LanguageSync = memo(() => {
    const { language } = useSettings();
    useLayoutEffect(() => {
        applyLanguage(language).catch(log.error);
    }, [language]);
    return null;
});

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
            supplier: async () => {
                const saved = await ipcRenderer.invoke('get-settings');
                // A saved language loads before the editor shows, so it does not flash in English.
                await applyLanguage(saved.language).catch(log.error);
                return saved;
            },
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
                                <LanguageSync />
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
