import { useColorMode } from '@chakra-ui/react';
import React, { SetStateAction, memo, useCallback, useEffect, useRef, useState } from 'react';
import { createContext } from 'use-context-selector';
import { log } from '../../common/log';
import { ChainnerSettings, defaultSettings } from '../../common/settings/settings';
import { noop } from '../../common/util';
import { useMemoObject } from '../hooks/useMemo';
import { ipcRenderer } from '../safeIpc';

interface SettingsContextValue {
    settings: Readonly<ChainnerSettings>;
    updateSettings: (
        update:
            | Partial<Readonly<ChainnerSettings>>
            | ((prev: Readonly<ChainnerSettings>) => Partial<Readonly<ChainnerSettings>>)
    ) => void;
    setSetting: <K extends keyof ChainnerSettings>(
        key: K,
        update: SetStateAction<ChainnerSettings[K]>
    ) => void;
}

export const SettingsContext = createContext<Readonly<SettingsContextValue>>({
    settings: defaultSettings,
    updateSettings: noop,
    setSetting: noop,
});

export const SettingsProvider = memo(
    ({
        initialSettings,
        children,
    }: React.PropsWithChildren<{ initialSettings: ChainnerSettings }>) => {
        const [settings, setSettings] = useState<ChainnerSettings>(() => ({
            ...defaultSettings,
            ...initialSettings,
        }));

        // setters that change nothing keep the settings object, so nothing re-renders or is saved
        const updateSettings: SettingsContextValue['updateSettings'] = useCallback((update) => {
            setSettings((prev) => {
                const changes = typeof update === 'function' ? update(prev) : update;
                const keys = Object.keys(changes) as (keyof ChainnerSettings)[];
                if (keys.every((key) => prev[key] === changes[key])) return prev;
                return { ...prev, ...changes };
            });
        }, []);

        const setSetting: SettingsContextValue['setSetting'] = useCallback((key, update) => {
            setSettings((prev) => {
                const oldValue = prev[key];
                const newValue = typeof update === 'function' ? update(oldValue) : update;
                if (oldValue === newValue) return prev;
                return { ...prev, [key]: newValue };
            });
        }, []);

        const savedSettingsRef = useRef(settings);
        useEffect(() => {
            if (savedSettingsRef.current === settings) return;
            savedSettingsRef.current = settings;
            ipcRenderer.invoke('set-settings', settings).catch(log.error);
        }, [settings]);

        // update theme
        const { setColorMode } = useColorMode();
        useEffect(() => {
            const [, darkOrLight] = settings.theme.split('-');
            setColorMode(darkOrLight);
            document.documentElement.setAttribute('data-custom-theme', settings.theme);
        }, [setColorMode, settings.theme]);

        const contextValue = useMemoObject<SettingsContextValue>({
            settings,
            updateSettings,
            setSetting,
        });

        return <SettingsContext.Provider value={contextValue}>{children}</SettingsContext.Provider>;
    }
);
