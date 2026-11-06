import i18n from 'i18next';
import TimeAgo, { LocaleData } from 'javascript-time-ago';
import { initReactI18next } from 'react-i18next';
import { DEFAULT_OPTIONS, activateLanguage } from '../common/i18n';
import { log } from '../common/log';

i18n.use(initReactI18next).init(DEFAULT_OPTIONS).catch(log.error);

// One per UI language other than English; pt-BR and zh-CN resolve to pt and zh.
const timeAgoLocales = new Map<string, () => Promise<{ default: LocaleData }>>([
    ['es', () => import('javascript-time-ago/locale/es.json')],
    ['de', () => import('javascript-time-ago/locale/de.json')],
    ['fr', () => import('javascript-time-ago/locale/fr.json')],
    ['it', () => import('javascript-time-ago/locale/it.json')],
    ['ja', () => import('javascript-time-ago/locale/ja.json')],
    ['zh-CN', () => import('javascript-time-ago/locale/zh.json')],
    ['ru', () => import('javascript-time-ago/locale/ru.json')],
    ['ko', () => import('javascript-time-ago/locale/ko.json')],
    ['nl', () => import('javascript-time-ago/locale/nl.json')],
    ['pl', () => import('javascript-time-ago/locale/pl.json')],
    ['tr', () => import('javascript-time-ago/locale/tr.json')],
    ['uk', () => import('javascript-time-ago/locale/uk.json')],
    ['hi', () => import('javascript-time-ago/locale/hi.json')],
    ['pt-BR', () => import('javascript-time-ago/locale/pt.json')],
]);

const loadTimeAgoLocale = async (language: string): Promise<void> => {
    const load = timeAgoLocales.get(language);
    if (load) {
        TimeAgo.addLocale((await load()).default);
    }
};

/**
 * Switches the UI, and the relative times it shows, to a saved language setting.
 */
export const applyLanguage = (language: string | undefined): Promise<void> =>
    activateLanguage(language, loadTimeAgoLocale);
