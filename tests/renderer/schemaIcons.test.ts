import { readFileSync, readdirSync } from 'fs';
import path from 'path';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { IconType } from 'react-icons';
import * as bs from 'react-icons/bs';
import * as cg from 'react-icons/cg';
import * as fa from 'react-icons/fa';
import * as gi from 'react-icons/gi';
import * as im from 'react-icons/im';
import * as md from 'react-icons/md';
import { describe, expect, test } from 'vitest';
import { IconFactory } from '../../src/renderer/components/CustomIcons';

const packs: Record<string, Partial<Record<string, IconType>>> = { bs, cg, fa, gi, im, md };

const backendIconNames = (): string[] => {
    const names = new Set<string>();
    const backend = path.join(__dirname, '../../backend/src');
    for (const entry of readdirSync(backend, { recursive: true, withFileTypes: true })) {
        if (entry.isFile() && entry.name.endsWith('.py')) {
            const source = readFileSync(path.join(entry.parentPath, entry.name), 'utf8');
            for (const [, name] of source.matchAll(/["']((?:Bs|Cg|Fa|Gi|Im|Md)[A-Z]\w+)["']/g)) {
                names.add(name);
            }
        }
    }
    return [...names].sort();
};

const drawing = (markup: string): string => markup.replace(/^.*?<svg[^>]*>/s, '');

describe('schema icons', () => {
    const names = backendIconNames();

    test('the scan of the backend finds its icons', () => {
        expect(names).toContain('BsQuestionCircleFill');
        expect(names.length).toBeGreaterThan(50);
    });

    test.each(names)('%s draws at once, as its pack draws it', (name) => {
        const expected = packs[name.slice(0, 2).toLowerCase()][name];

        const markup = renderToStaticMarkup(createElement(IconFactory, { icon: name }));

        expect(expected).toBeDefined();
        expect(drawing(markup)).toBe(drawing(renderToStaticMarkup(createElement(expected!))));
    });

    test('an icon no schema names draws nothing until its pack has loaded', () => {
        const markup = renderToStaticMarkup(createElement(IconFactory, { icon: 'MdAccessAlarm' }));

        expect(markup).not.toContain('<svg');
    });

    test.each(['bs', 'cg', 'im', 'md'])(
        'the %s pack loaded on first use draws every icon as the ES build does',
        (pack) => {
            const drawings = (file: string) => {
                const source = readFileSync(
                    path.join(__dirname, '../../node_modules/react-icons', pack, file),
                    'utf8'
                );
                return [...source.matchAll(/function (\w+) \(props\) \{\s*return GenIcon\((.*)\)/g)]
                    .map(([, name, icon]) => `${name} ${icon}`)
                    .sort();
            };

            const esm = drawings('index.esm.js');

            expect(esm.length).toBeGreaterThan(100);
            expect(drawings('index.js')).toEqual(esm);
        }
    );
});
