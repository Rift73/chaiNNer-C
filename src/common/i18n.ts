import i18n, { InitOptions } from 'i18next';
import en from './locales/en/translation.json';

const resources = {
    en: { translation: en },
};

// English is always loaded. The others load on first use, so they add nothing to startup.
const otherLanguages = new Map<string, () => Promise<{ default: object }>>([
    ['es', () => import('./locales/es.json')],
    ['de', () => import('./locales/de.json')],
    ['fr', () => import('./locales/fr.json')],
    ['it', () => import('./locales/it.json')],
    ['ja', () => import('./locales/ja.json')],
    ['zh-CN', () => import('./locales/zh-CN.json')],
    ['ru', () => import('./locales/ru.json')],
    ['ko', () => import('./locales/ko.json')],
    ['nl', () => import('./locales/nl.json')],
    ['pl', () => import('./locales/pl.json')],
    ['tr', () => import('./locales/tr.json')],
    ['uk', () => import('./locales/uk.json')],
    ['hi', () => import('./locales/hi.json')],
    ['pt-BR', () => import('./locales/pt-BR.json')],
]);

/**
 * Maps a saved language setting to a supported language. Anything else is English.
 */
export const normalizeLanguage = (language: string | undefined): string => {
    return language !== undefined && otherLanguages.has(language) ? language : 'en';
};

let wantedLanguage = 'en';

/**
 * Switches the i18next instance to a saved language setting, loading the language if needed, and
 * first what `prepare` loads for it. The last call wins, whichever load finishes first.
 */
export const activateLanguage = async (
    language: string | undefined,
    prepare?: (target: string) => Promise<void>
): Promise<void> => {
    const target = normalizeLanguage(language);
    wantedLanguage = target;
    if (target === i18n.language) {
        return;
    }
    await prepare?.(target);
    if (!i18n.hasResourceBundle(target, 'translation')) {
        const { default: translation } = await otherLanguages.get(target)!();
        i18n.addResourceBundle(target, 'translation', translation);
    }
    // a later call may have asked for another language while this one loaded
    if (target === wantedLanguage) {
        await i18n.changeLanguage(target);
    }
};

export const DEFAULT_OPTIONS: InitOptions = {
    resources,
    lng: 'en',
    fallbackLng: 'en',

    interpolation: {
        escapeValue: false,
    },

    returnNull: false,

    // Enable pluralization
    pluralSeparator: '_',
    contextSeparator: '_',
};
