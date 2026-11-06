import { readFileSync, readdirSync } from 'fs';
import i18n from 'i18next';
import path from 'path';
import ts from 'typescript';
import { beforeAll, expect, test } from 'vitest';
import { DEFAULT_OPTIONS, activateLanguage, normalizeLanguage } from '../../src/common/i18n';
import { defaultSettings } from '../../src/common/settings/settings';

const SRC = path.join(__dirname, '../../src');
const LOCALES = path.join(SRC, 'common/locales');
const PLURAL_SUFFIX = /_(?:zero|one|two|few|many|other)$/;

type Translations = { [key: string]: string | Translations };

const flatten = (translations: Translations, prefix = ''): Map<string, string> => {
    const flat = new Map<string, string>();
    for (const [key, value] of Object.entries(translations)) {
        if (typeof value === 'string') {
            flat.set(prefix + key, value);
        } else {
            flatten(value, `${prefix}${key}.`).forEach((v, k) => flat.set(k, v));
        }
    }
    return flat;
};

const readLocale = (file: string) =>
    JSON.parse(readFileSync(path.join(LOCALES, file), 'utf-8')) as Translations;

const english = flatten(readLocale('en/translation.json'));
const languageFiles = readdirSync(LOCALES).filter((name) => name.endsWith('.json'));

/** The English values a key can show: its own, or its plural forms. */
const englishValues = (key: string): string[] =>
    [...english]
        .filter(([k]) => k === key || k.replace(PLURAL_SUFFIX, '') === key)
        .map(([, v]) => v);

interface Use {
    readonly where: string;
    readonly key: string;
    readonly text?: string;
}

const stringOf = (node: ts.Node | undefined): string | undefined =>
    node && (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node))
        ? node.text
        : undefined;

const readTrans = (element: ts.JsxOpeningLikeElement, where: string): Use | undefined => {
    const attribute = (name: string) =>
        element.attributes.properties.find(
            (p): p is ts.JsxAttribute => ts.isJsxAttribute(p) && p.name.getText() === name
        )?.initializer;
    const key = stringOf(attribute('i18nKey'));
    return key === undefined ? undefined : { where, key, text: stringOf(attribute('defaults')) };
};

/** Every t() call and Trans element of a source file; a key built at run time fails the test. */
const readUses = (file: string): Use[] => {
    const code = readFileSync(file, 'utf-8');
    const source = ts.createSourceFile(file, code, ts.ScriptTarget.Latest, true);
    const uses: Use[] = [];
    const visit = (node: ts.Node) => {
        const where = `${path.relative(SRC, file)}:${
            source.getLineAndCharacterOfPosition(node.getStart()).line + 1
        }`;
        if (ts.isCallExpression(node)) {
            const callee = node.expression;
            const isT =
                (ts.isIdentifier(callee) && callee.text === 't') ||
                (ts.isPropertyAccessExpression(callee) && callee.name.text === 't');
            if (isT) {
                const key = stringOf(node.arguments[0]);
                if (key === undefined) {
                    throw new Error(`${where}: the key of t() is not a literal`);
                }
                uses.push({ where, key, text: stringOf(node.arguments[1]) });
            }
        }
        if (
            (ts.isJsxSelfClosingElement(node) || ts.isJsxOpeningElement(node)) &&
            node.tagName.getText() === 'Trans'
        ) {
            const use = readTrans(node, where);
            if (use) uses.push(use);
        }
        // ConditionExplanation's sentences are Trans elements with the name as the last key part
        if (ts.isVariableDeclaration(node) && node.name.getText() === 'SENTENCES') {
            const sentences = (node.initializer as ts.AsExpression)
                .expression as ts.ObjectLiteralExpression;
            for (const sentence of sentences.properties as readonly ts.PropertyAssignment[]) {
                uses.push({
                    where,
                    key: `nodeDocumentation.conditionExplanation.${sentence.name.getText()}`,
                    text: stringOf(sentence.initializer),
                });
            }
        }
        ts.forEachChild(node, visit);
    };
    visit(source);
    return uses;
};

const sourceFiles = readdirSync(SRC, { recursive: true, encoding: 'utf-8' })
    .filter((name) => /\.tsx?$/.test(name))
    .map((name) => path.join(SRC, name));
const uses = sourceFiles.flatMap(readUses);

beforeAll(async () => {
    await i18n.init(DEFAULT_OPTIONS);
});

test('the source has translated strings', () => {
    expect(uses.length).toBeGreaterThan(300);
});

test('every key used in the source exists in English', () => {
    expect(uses.filter((use) => englishValues(use.key).length === 0)).toEqual([]);
});

test('the text in the source is the English value', () => {
    const differing = uses.filter(
        (use) => use.text !== undefined && !englishValues(use.key).includes(use.text)
    );
    expect(differing).toEqual([]);
});

test.each(languageFiles)('%s parses and has no key absent from English', (file) => {
    const absent = [...flatten(readLocale(file)).keys()].filter(
        (key) => !english.has(key) && englishValues(key.replace(PLURAL_SUFFIX, '')).length === 0
    );
    expect(absent).toEqual([]);
});

test('English comes from its file', () => {
    expect(english.get('header.run')).toBe('Run');
    expect(i18n.t('header.run')).toBe('Run');
});

test('a count keeps the sentence it always had in English', () => {
    expect(i18n.t('dependencyManager.updatesAvailableDescription', { count: 1 })).toBe(
        'There are 1 packages with available dependency updates.'
    );
    expect(i18n.t('dependencyManager.updatesAvailableDescription', { count: 2 })).toBe(
        'There are 2 packages with available dependency updates.'
    );
    expect(i18n.t('chain.invalidConnections', { count: 1 })).toContain(
        'The chain contains 1 connection(s)'
    );
    expect(i18n.t('dependencyManager.package', { count: 1 })).toBe('package');
    expect(i18n.t('dependencyManager.package', { count: 0 })).toBe('packages');
});

test('every language file can be chosen', () => {
    expect(languageFiles.length).toBe(14);
    for (const file of languageFiles) {
        const language = file.replace(/\.json$/, '');
        expect(normalizeLanguage(language)).toBe(language);
    }
});

test('an unknown language is English', () => {
    for (const language of [undefined, '', 'xx', 'pt', 'constructor']) {
        expect(normalizeLanguage(language)).toBe('en');
    }
});

test('English is the default on every install', () => {
    expect(defaultSettings.language).toBe('en');
    expect(i18n.language).toBe('en');
});

test('a language loads on first use, and what it lacks shows in English', async () => {
    const german = flatten(readLocale('de.json'));
    expect(i18n.hasResourceBundle('de', 'translation')).toBe(false);

    await activateLanguage('de');
    expect(i18n.language).toBe('de');
    expect(i18n.t('header.run')).toBe(german.get('header.run'));
    expect(german.has('inputs.videoSeek.runThisFrame')).toBe(false);
    expect(i18n.t('inputs.videoSeek.runThisFrame')).toBe('Run this frame');

    await activateLanguage('en');
    expect(i18n.t('header.run')).toBe('Run');
});

test('the last language asked for wins, even if an earlier one is still loading', async () => {
    await Promise.all([activateLanguage('ru'), activateLanguage('en')]);
    expect(i18n.language).toBe('en');

    await Promise.all([activateLanguage('ja'), activateLanguage('ko')]);
    expect(i18n.language).toBe('ko');

    await activateLanguage('en');
});
