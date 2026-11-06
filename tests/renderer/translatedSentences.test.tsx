import i18n from 'i18next';
import TimeAgo from 'javascript-time-ago';
import { renderToStaticMarkup } from 'react-dom/server';
import { Trans } from 'react-i18next';
import { expect, test } from 'vitest';
import { TransElement } from '../../src/renderer/components/TransElement';
import { applyLanguage } from '../../src/renderer/i18n';

const render = (
    i18nKey: string,
    components: Record<string, JSX.Element>,
    values?: Record<string, string>
) =>
    renderToStaticMarkup(
        <Trans
            components={components}
            i18nKey={i18nKey}
            values={values}
        />
    );

test('the favorites hints keep their italics and icon', () => {
    expect(render('favorites.addFavoritesTooltip', { em: <em /> })).toBe(
        'Add Favorites by right-clicking nodes and selecting <em>Add to Favorites</em>.'
    );
    expect(
        render('favorites.addFavoritesTooltipExpanded', {
            em: <em />,
            star: <TransElement element={<svg />} />,
        })
    ).toBe(
        'Add Favorites by hovering over nodes and clicking the <svg></svg> icon, or by right-clicking and selecting <em>Add to Favorites</em>.'
    );
});

test('the install hint keeps its italics', () => {
    expect(render('nodeSelector.clickToInstall', { em: <em /> }, { packageName: 'PyTorch' })).toBe(
        '<em>Click</em> to open the dependency manager to install PyTorch.'
    );
});

test.each([
    ['enumIs', '<b>Mode</b> is <i>x</i>.'],
    ['enumIsNot', '<b>Mode</b> is not <i>x</i>.'],
    ['enumIsOneOf', '<b>Mode</b> is one of <i>x</i>.'],
    ['enumIsNotOneOf', '<b>Mode</b> is not one of <i>x</i>.'],
    ['typeIsOfType', '<b>Mode</b> is of type <i>x</i>.'],
    ['typeIsNotOfType', '<b>Mode</b> is not of type <i>x</i>.'],
    ['typeIs', '<b>Mode</b> is an image.'],
    ['typeIsNot', '<b>Mode</b> is not an image.'],
    ['iteratedBy', '<b>Mode</b> is iterated by <b>Load</b>.'],
    ['notIteratedBy', '<b>Mode</b> is not iterated by <b>Load</b>.'],
])('the condition sentence %s reads as before', (sentence, expected) => {
    const element = <TransElement element={<i>x</i>} />;
    expect(
        render(
            `nodeDocumentation.conditionExplanation.${sentence}`,
            { name: <b />, options: element, type: element },
            { name: 'Mode', type: 'an image', source: 'Load' }
        )
    ).toBe(expected);
});

test('a translation without the italics renders too', async () => {
    await applyLanguage('de');
    expect(render('favorites.addFavoritesTooltip', { em: <em /> })).not.toContain('<em>');
    expect(render('favorites.addFavoritesTooltip', { em: <em /> })).not.toBe(
        'Add Favorites by right-clicking nodes and selecting <em>Add to Favorites</em>.'
    );
    await applyLanguage('en');
});

test('the last language asked for wins while an earlier one loads', async () => {
    await Promise.all([applyLanguage('es'), applyLanguage('en')]);
    expect(i18n.language).toBe('en');

    await Promise.all([applyLanguage('ja'), applyLanguage('ko')]);
    expect(i18n.language).toBe('ko');

    await applyLanguage('en');
});

test('relative times are in the language once it applies', async () => {
    await applyLanguage('fr');
    expect(new TimeAgo('fr').format(Date.now() - 120_000)).toContain('il y a');
    await applyLanguage('en');
});
