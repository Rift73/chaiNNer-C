/* Shared inputs and PASS/FAIL reporting for the bundle-only verify_*.cjs tools.
 * Reads the installed app and the package named by the package manifest
 * (identity.installed_app, identity.destination; the v2 manifest) or by
 * the --installed-app/--package overrides. Writes nothing. */
'use strict';

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { parseArgs } = require('node:util');

const RENDERER = 'resources/app/.vite/renderer/main_window/index.js';
const MAIN = 'resources/app/.vite/build/main.js';
const CSS = 'resources/app/.vite/renderer/main_window/index.css';
// The minified names, stubs and anchors in the verify_*.cjs tools were reviewed
// against exactly these installed bundles (independent_ui.BASELINE_HASHES).
const INSTALLED_SHA256 = {
    [RENDERER]: '50f22a101c16c97fd1db413e3e0becf647c360b817f09b8a3089fa649fd16877',
    [MAIN]: 'df979a26746d814f4dd8e4ce555f6114304a0b7f72cef429de783c646ad26f10',
    [CSS]: '2b33220c0429e43d9a11cea32cc32b5df6fb919d3cc36be7d074f8e5609c9f81',
};

function sha256(value) {
    return crypto.createHash('sha256').update(value).digest('hex');
}

function count(text, needle) {
    return text.split(needle).length - 1;
}

function region(text, start, end) {
    const left = text.indexOf(start);
    assert.notEqual(left, -1, `Missing reviewed start: ${start}`);
    assert.equal(text.indexOf(start, left + 1), -1, `Ambiguous reviewed start: ${start}`);
    const right = text.indexOf(end, left + start.length);
    assert.notEqual(right, -1, `Missing reviewed end: ${end}`);
    return { text: text.slice(left, right), left, right };
}

function resolveRoots() {
    const { values } = parseArgs({
        options: {
            manifest: { type: 'string', default: path.resolve(__dirname, '../../out/chaiNNer-C/chainner-c-package.json') },
            'installed-app': { type: 'string' },
            package: { type: 'string' },
        },
    });
    let installedApp = values['installed-app'];
    let packageRoot = values.package;
    if (installedApp === undefined || packageRoot === undefined) {
        const { identity } = JSON.parse(fs.readFileSync(values.manifest, 'utf8'));
        assert.equal(typeof identity?.installed_app, 'string', `${values.manifest}: identity.installed_app missing`);
        assert.equal(typeof identity?.destination, 'string', `${values.manifest}: identity.destination missing`);
        installedApp ??= identity.installed_app;
        packageRoot ??= identity.destination;
    }
    return { installedApp: path.resolve(installedApp), packageRoot: path.resolve(packageRoot) };
}

async function loadBundles(check, relatives) {
    const roots = resolveRoots();
    console.log(`installed app: ${roots.installedApp}`);
    console.log(`package:       ${roots.packageRoot}`);
    const installed = {};
    const pinned = await check('installed bundles match the reviewed baseline pins', () => {
        for (const relative of relatives) {
            const bytes = fs.readFileSync(path.join(roots.installedApp, relative));
            assert.equal(sha256(bytes), INSTALLED_SHA256[relative], `Unreviewed installed bundle: ${relative}`);
            installed[relative] = bytes.toString('utf8');
        }
    });
    if (!pinned) return undefined;
    const packaged = {};
    for (const relative of relatives) {
        const bytes = fs.readFileSync(path.join(roots.packageRoot, relative));
        console.log(`packaged ${relative} sha256 ${sha256(bytes)}`);
        packaged[relative] = bytes.toString('utf8');
    }
    return { roots, installed, packaged };
}

async function runChecks(title, body) {
    let passed = 0;
    let failed = 0;
    async function check(name, action) {
        try {
            await action();
        } catch (error) {
            failed += 1;
            console.log(`FAIL ${name}\n    ${String(error?.message ?? error).replaceAll('\n', '\n    ')}`);
            return false;
        }
        passed += 1;
        console.log(`PASS ${name}`);
        return true;
    }
    try {
        await body(check);
    } catch (error) {
        failed += 1;
        console.log(`FAIL ${title} stopped: ${error?.stack ?? error}`);
    }
    const ok = failed === 0 && passed > 0;
    console.log(`${ok ? 'PASS' : 'FAIL'} ${title}: ${passed} passed, ${failed} failed`);
    process.exitCode = ok ? 0 : 1;
}

module.exports = { CSS, INSTALLED_SHA256, MAIN, RENDERER, count, loadBundles, region, runChecks, sha256 };
