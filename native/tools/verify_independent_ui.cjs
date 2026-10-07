/* Offline regression checks for the independent edition's packaged bundles
 * (independent_ui.py + input-drop.json) against the pinned installed bundles.
 * No Electron startup, network access, GPU work, dependency installation or
 * file writes. VM element trees establish behavior/structure, not rendering.
 * Usage: node verify_independent_ui.cjs [--manifest <file>] [--installed-app <dir>] [--package <dir>]
 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const vm = require('node:vm');
const { CSS, INSTALLED_SHA256, MAIN, RENDERER, count, loadBundles, region, runChecks, sha256 } = require('./bundle_verify_common.cjs');

const FORBIDDEN_RENDERER = ['api.github.com', '/releases/latest', '/releases?per_page=', 'Update Available (', 'th("checkForUpdatesOnStartup")'];
// The product version shown in the header (independent_ui.PRODUCT_VERSION); the
// package's own package.json keeps the upstream baseline version.
const PRODUCT_VERSION = '0.3.0';
// Upstream keeps the integrated Python only when its version equals the 3.11.5
// download table (xrA); the port keeps any 3.14.0 or newer. s is the probed
// Python, I the table version, gR the bundled semver.
const PYTHON_CHECK_UPSTREAM = 'if(gR.eq(s.version,I))return s}';
const PYTHON_CHECK_PORT = 'if(gR.gte(s.version,"3.14.0"))return s}';
// independent_ui.MAIN_EDITS, restated so this check does not trust the patcher.
// Consult 2 Q5: shown versions are chaiNNer-C's, app.getVersion() (xf) stays
// upstream's; Q6: the integrated-Python redownload branch throws instead.
// The pinned app's package.json version, checked below against both package.json files.
const UPSTREAM_VERSION = '0.25.1-nightly.2025-10-21';
const REDOWNLOAD_ERROR = "chaiNNer-C's integrated Python is missing or older than 3.14; reinstall chaiNNer-C";
const MAIN_EDITS = [
    {
        label: 'About message',
        before: 'message:`chaiNNer ${Wg.app.getVersion()}`',
        after: `message:"chaiNNer v${PRODUCT_VERSION}"`,
    },
    {
        label: 'startup log line',
        before: '_g.info(`chaiNNer Version: ${xf}`)',
        after: `_g.info("chaiNNer-C ${PRODUCT_VERSION} (upstream ${UPSTREAM_VERSION})")`,
    },
    {
        label: 'system information',
        before: 'app:{version:Wg.app.getVersion(),packaged:',
        after: `app:{version:"${PRODUCT_VERSION}",upstream:"${UPSTREAM_VERSION}",packaged:`,
    },
    {
        label: 'Release Notes item removed',
        before: '...xs?[]:[{label:"About chaiNNer",click:C}],{label:"Release Notes",click:async()=>{await Wg.shell.openExternal(`https://github.com/chaiNNer-org/chaiNNer/releases/tag/v${Wg.app.getVersion()}`)}},',
        after: '...xs?[]:[{label:"About chaiNNer",click:C}],',
    },
    {
        label: 'integrated-Python redownload branch closed',
        before: 'const i=eB.resolve(eB.join(A,"/python"));await pe.rm(i,{recursive:!0,force:!0}),_g.info(`Integrated Python not found at ${C}`);const o="python.tar.gz",n=eB.join(A,o);if(_g.info("Downloading integrated Python..."),g(0,"download"),await new YrA({url:Q,directory:A,fileName:o,cloneFiles:!1,onProgress:s=>g(Number(s),"download")}).download(),_g.info("Extracting integrated Python..."),g(0,"extract"),await JrA(A,n,s=>g(s,"extract")),_g.info("Removing downloaded files..."),await pe.rm(n),B==="linux"||B==="darwin"){_g.info("Granting permissions for integrated python...");try{await pe.chmod(C,4095)}catch(s){_g.warn(s)}}return qU([C])}',
        after: `throw new Error(${JSON.stringify(REDOWNLOAD_ERROR)})}`,
    },
];

function maskRegion(text, start, end, label) {
    const bounds = region(text, start, end);
    return text.slice(0, bounds.left) + `<${label}>` + text.slice(bounds.right);
}

function run(code, sandbox, filename) {
    const context = vm.createContext(sandbox);
    new vm.Script(code, { filename }).runInContext(context, { timeout: 2000 });
    return sandbox;
}

function canonical(value) {
    if (typeof value === 'function') return '[function]';
    if (Array.isArray(value)) {
        return Array.from(value)
            .filter((item) => item !== undefined && item !== null && item !== false)
            .map(canonical);
    }
    if (value && typeof value === 'object') {
        return Object.fromEntries(
            Object.entries(value)
                .filter(([, item]) => item !== undefined)
                .map(([key, item]) => [key, canonical(item)])
        );
    }
    return value;
}

function element(type, props = {}) {
    return { type, props };
}

const uiSymbols = {
    j2: 'HStack', E_: 'Image', Fv: 'Heading', EL: 'Tag', i5: 'Tooltip',
    ms: 'IconButton', K11: 'DownloadIcon', zZ: 'Modal', Ik: 'ModalOverlay',
    wZ: 'ModalContent', Rk: 'ModalHeader', CZ: 'ModalCloseButton', Pk: 'ModalBody',
    K8: 'Markdown', D_: 'ModalFooter', Ia: 'Button', j11: 'Link',
    XGt: 'AlertDialog', YGt: 'AlertDialogContent', m4: 'VStack', O_: 'Divider',
    Fb: 'SettingsToggle',
};

function uiSandbox(version) {
    const effects = [];
    const requests = [];
    const settingReads = [];
    const noop = () => {};
    const react = {
        memo: (component) => component,
        useEffect: (effect) => effects.push(effect),
        useState: (initial) => [initial, noop],
        useRef: (current) => ({ current }),
    };
    const jsx = { jsx: element, jsxs: element, Fragment: 'Fragment' };
    const sandbox = {
        ...uiSymbols,
        W: react,
        _: jsx,
        VTt: version,
        elr: '[unchanged logo asset]',
        tg: () => ({ checkForUpdatesOnStartup: false }),
        Qb: () => ({ isOpen: false, onOpen: noop, onClose: noop }),
        nd1: () => [undefined, noop],
        th: (key) => {
            settingReads.push(key);
            return [false, noop];
        },
        gJ: false,
        d01: { gt: () => false, lte: () => true },
        fetch: async (url) => {
            requests.push(String(url));
            return { ok: true, json: async () => ({ tag_name: version }) };
        },
    };
    return { sandbox, effects, requests, settingReads };
}

async function evaluateComponent(declaration, name, version, filename) {
    const test = uiSandbox(version);
    run(`const ${declaration.replace(/^,/, '')}; globalThis.tree = ${name}();`, test.sandbox, filename);
    for (const effect of test.effects) await effect();
    // Finish the bounded promise chain from the installed release fetch stub.
    for (let i = 0; i < 8; i += 1) await Promise.resolve();
    return { ...test, tree: canonical(test.sandbox.tree) };
}

function firstElement(tree, type) {
    if (!tree || typeof tree !== 'object') return undefined;
    if (tree.type === type) return tree;
    for (const child of Object.values(tree)) {
        if (Array.isArray(child)) {
            for (const item of child) {
                const found = firstElement(item, type);
                if (found) return found;
            }
        } else {
            const found = firstElement(child, type);
            if (found) return found;
        }
    }
    return undefined;
}

function syntaxCheck(file, source, module) {
    const args = module ? ['--input-type=module', '--check'] : ['--check', file];
    const result = spawnSync(process.execPath, args, {
        input: module ? source : undefined,
        encoding: 'utf8',
        windowsHide: true,
        maxBuffer: 1024 * 1024,
    });
    if (result.error) throw result.error;
    assert.equal(result.status, 0, `${file}: ${result.stderr}`);
}

// The integrated-Python setup function (xrA) run with stubs for a runtime that
// exists or not and reports a version: what it returns or throws, and every
// probe, delete, download and extraction it attempts.
async function integratedPython(mainBundle, filename, exists, version) {
    const declaration = region(mainBundle, 'xrA=async(A,g)=>{', ',qrA=wD(').text;
    const calls = [];
    const compare = (a, b) => {
        const [x, y] = [a, b].map((text) => text.split('.').map(Number));
        return x.map((part, i) => part - y[i]).find((difference) => difference !== 0) ?? 0;
    };
    const sandbox = {
        A5: () => 'win32',
        QP: { win32: { url: 'https://example.invalid/python.tar.gz', version: '3.11.5' } },
        IP: (root) => `${root}/python/python.exe`,
        KrA: async () => exists,
        qU: async (paths) => {
            calls.push(['probe', ...paths]);
            return { version };
        },
        gR: { eq: (a, b) => compare(a, b) === 0, gte: (a, b) => compare(a, b) >= 0 },
        eB: { resolve: (target) => target, join: (...parts) => parts.join('/') },
        pe: { rm: async (target) => calls.push(['rm', target]), chmod: async () => calls.push(['chmod']) },
        YrA: class {
            constructor(options) {
                calls.push(['download', options.url]);
            }
            async download() {}
        },
        JrA: async () => calls.push(['extract']),
        _g: { info: () => {}, warn: () => {} },
    };
    run(`const ${declaration}; globalThis.setup = xrA;`, sandbox, filename);
    try {
        return { python: await sandbox.setup('C:/package', () => {}), calls };
    } catch (error) {
        return { error: error.message, calls };
    }
}

function settingsModule(mainBundle, filename) {
    const declaration = region(mainBundle, 'const pH=', ',rY=eB.join(').text;
    return run(`${declaration}; globalThis.result = {defaultSettings:pH,migrateOldStorageSettings:xuA,migrateSettings:y9};`, {}, filename).result;
}

function legacyStorage(reads) {
    return {
        keys: ['check-upd-on-strtup-2', 'theme', 'snap-to-grid'],
        getItem(key) {
            reads.push(key);
            return {
                'check-upd-on-strtup-2': 'true',
                theme: '"light"',
                'snap-to-grid': 'true',
            }[key] ?? null;
        },
    };
}

function checkSettings({ defaultSettings, migrateSettings, migrateOldStorageSettings }) {
    assert.equal(defaultSettings.checkForUpdatesOnStartup, false);
    for (const enabled of [undefined, true, false]) {
        const source = { theme: 'default-light', snapToGrid: true };
        if (enabled !== undefined) source.checkForUpdatesOnStartup = enabled;
        const result = migrateSettings(source);
        assert.equal(result.checkForUpdatesOnStartup, false);
        assert.equal(result.theme, 'default-light');
        assert.equal(result.snapToGrid, true);
    }
    const reads = [];
    const legacy = migrateOldStorageSettings(legacyStorage(reads));
    assert.equal(legacy.checkForUpdatesOnStartup, false);
    assert.equal(reads.includes('check-upd-on-strtup-2'), false);
    const migrated = migrateSettings(legacy);
    assert.equal(migrated.checkForUpdatesOnStartup, false);
    assert.equal(migrated.theme, 'default-light');
    assert.equal(migrated.snapToGrid, true);
}

runChecks('verify_independent_ui', async (check) => {
    const bundles = await loadBundles(check, [RENDERER, MAIN, CSS]);
    if (!bundles) return;
    const { roots } = bundles;
    const originalRenderer = bundles.installed[RENDERER];
    const originalMain = bundles.installed[MAIN];
    const renderer = bundles.packaged[RENDERER];
    const mainBundle = bundles.packaged[MAIN];
    const packageJson = JSON.parse(fs.readFileSync(path.join(roots.packageRoot, 'resources/app/package.json'), 'utf8'));
    const dropEditsFile = path.join(__dirname, 'input-drop.json');
    const dropEdits = JSON.parse(fs.readFileSync(dropEditsFile, 'utf8'));
    console.log(`input-drop.json sha256 ${sha256(fs.readFileSync(dropEditsFile))}`);
    const trtFile = path.join(__dirname, 'tensorrt-types.json');
    const trt = JSON.parse(fs.readFileSync(trtFile, 'utf8'));
    console.log(`tensorrt-types.json sha256 ${sha256(fs.readFileSync(trtFile))}`);
    const trtClearFile = path.join(__dirname, 'tensorrt-clear.json');
    const trtClear = JSON.parse(fs.readFileSync(trtClearFile, 'utf8'));
    console.log(`tensorrt-clear.json sha256 ${sha256(fs.readFileSync(trtClearFile))}`);
    // Both TensorRT patches' edits in the order independent_ui applies them.
    const trtEdits = [...trt.edits, ...trtClear.edits];
    // Undo the TensorRT edits of one bundle, last first, each after exactly once.
    function revertTrt(text, bundle) {
        for (const edit of trtEdits.filter((e) => e.bundle === bundle).reverse()) {
            assert.equal(count(text, edit.after), 1, `Missing/duplicate TensorRT edit: ${edit.label}`);
            text = text.replace(edit.after, () => edit.before);
        }
        return text;
    }

    await check('package.json main is the reviewed main bundle', () => {
        assert.equal(path.posix.join('resources/app', packageJson.main), MAIN);
    });
    await check('input-drop.json is pinned to the installed renderer', () => {
        assert.equal(dropEdits.baseline_renderer_sha256, INSTALLED_SHA256[RENDERER]);
    });
    await check('tensorrt-types.json is pinned to the installed renderer, main and stylesheet', () => {
        assert.deepEqual(trt.baseline_sha256, INSTALLED_SHA256);
    });
    await check('tensorrt-clear.json is pinned to the installed renderer, main and stylesheet', () => {
        assert.deepEqual(trtClear.baseline_sha256, INSTALLED_SHA256);
    });
    await check('package carries no inert squirrel.exe installer helper', () => {
        assert.equal(fs.existsSync(path.join(roots.packageRoot, 'squirrel.exe')), false);
    });
    await check('packaged main bundle syntax (CommonJS file)', () => {
        syntaxCheck(path.join(roots.packageRoot, MAIN), mainBundle, false);
    });
    await check('packaged renderer bundle syntax (module via stdin)', () => {
        syntaxCheck(path.join(roots.packageRoot, RENDERER), renderer, true);
    });

    await check('header: no release fetch or update UI; logo and title kept, no Alpha badge, product version shown', async () => {
        const beforeHeader = region(originalRenderer, ',ln0="chaiNNer-org/chaiNNer"', ',clr=W.memo(').text;
        const afterHeader = region(renderer, ',llr=W.memo(', ',clr=W.memo(').text;
        const before = await evaluateComponent(beforeHeader, 'llr', packageJson.version, 'installed-AppInfo.js');
        const after = await evaluateComponent(afterHeader, 'llr', packageJson.version, 'packaged-AppInfo.js');
        assert.ok(before.effects.length > 0);
        assert.deepEqual(before.requests, ['https://api.github.com/repos/chaiNNer-org/chaiNNer/releases/latest']);
        assert.equal(after.effects.length, 0);
        assert.deepEqual(after.requests, []);
        const branding = firstElement(before.tree, 'HStack');
        const [logo, title, alpha, version] = branding.props.children;
        assert.deepEqual(branding.props.children.map((child) => child.type), ['Image', 'Heading', 'Tag', 'Tag']);
        assert.equal(alpha.props.children, 'Alpha');
        assert.deepEqual(version.props.children, ['v', packageJson.version]);
        assert.deepEqual(after.tree, {
            ...branding,
            props: { ...branding.props, children: [logo, title, { type: 'Tag', props: { children: `v${PRODUCT_VERSION}` } }] },
        });
        const shown = JSON.stringify(after.tree);
        assert.ok(!shown.includes('Alpha'), 'Alpha badge remains');
        assert.ok(!shown.includes(packageJson.version), 'upstream app version remains');
        assert.equal(count(originalRenderer, 'children:"Alpha"'), 1, 'installed: Alpha badge');
        assert.equal(count(renderer, 'children:"Alpha"'), 0, 'packaged: Alpha badge');
        assert.equal(count(renderer, `children:"v${PRODUCT_VERSION}"`), 1, 'packaged: product version');
    });

    await check('advanced settings: update toggle removed; other toggles unchanged', async () => {
        const beforeAdvanced = await evaluateComponent(region(originalRenderer, ',Yor=W.memo(', ',Qor=W.memo(').text, 'Yor', packageJson.version, 'installed-AdvancedSettings.js');
        const afterAdvanced = await evaluateComponent(region(renderer, ',Yor=W.memo(', ',Qor=W.memo(').text, 'Yor', packageJson.version, 'packaged-AdvancedSettings.js');
        const expectedAdvanced = canonical(beforeAdvanced.tree);
        expectedAdvanced.props.children = expectedAdvanced.props.children.filter((child) => child.props?.setting?.label !== 'Check for Update on Start-up');
        assert.notEqual(expectedAdvanced.props.children.length, canonical(beforeAdvanced.tree).props.children.length, 'Installed toggle not found');
        assert.deepEqual(afterAdvanced.tree, expectedAdvanced);
        assert.deepEqual(afterAdvanced.settingReads, beforeAdvanced.settingReads.filter((key) => key !== 'checkForUpdatesOnStartup'));
        assert.equal(afterAdvanced.effects.length, 0);
        assert.deepEqual(afterAdvanced.requests, []);
    });

    await check('installed settings baseline enables updates (makes the next check meaningful)', () => {
        const installed = settingsModule(originalMain, 'installed-settings.js');
        assert.equal(installed.defaultSettings.checkForUpdatesOnStartup, true);
        const reads = [];
        assert.equal(installed.migrateOldStorageSettings(legacyStorage(reads)).checkForUpdatesOnStartup, true);
        assert.ok(reads.includes('check-upd-on-strtup-2'));
    });
    await check('packaged settings: updates off for new, current and legacy profiles', () => {
        checkSettings(settingsModule(mainBundle, 'packaged-settings.js'));
    });

    await check('integrated Python: the 3.11.5 equality check is gone, the 3.14.0-or-newer check appears once', () => {
        assert.equal(count(originalMain, PYTHON_CHECK_UPSTREAM), 1, 'installed: equality check');
        assert.equal(count(originalMain, PYTHON_CHECK_PORT), 0, 'installed: 3.14.0 check');
        assert.equal(count(originalMain, 'gR.eq(s.version,'), 1, 'installed: any eq on the probed version');
        assert.equal(count(mainBundle, PYTHON_CHECK_UPSTREAM), 0, 'packaged: equality check');
        assert.equal(count(mainBundle, 'gR.eq(s.version,'), 0, 'packaged: any eq on the probed version');
        assert.equal(count(mainBundle, PYTHON_CHECK_PORT), 1, 'packaged: 3.14.0 check');
    });

    await check('main edits: each before once in installed, gone from packaged; each after once in packaged only', () => {
        for (const edit of MAIN_EDITS) {
            assert.equal(count(originalMain, edit.before), 1, `installed before: ${edit.label}`);
            // An after that is part of its before (the menu's kept context) is in installed only there.
            assert.equal(count(originalMain, edit.after), count(edit.before, edit.after), `installed after: ${edit.label}`);
            assert.equal(count(mainBundle, edit.before), 0, `packaged before: ${edit.label}`);
            assert.equal(count(mainBundle, edit.after), 1, `packaged after: ${edit.label}`);
        }
        for (const gone of ['Release Notes', 'releases/tag', 'chaiNNer Version: ', 'Downloading integrated Python', 'new YrA(', 'await pe.rm(i,{recursive:!0,force:!0})']) {
            assert.equal(count(originalMain, gone), 1, `installed: ${gone}`);
            assert.equal(count(mainBundle, gone), 0, `packaged: ${gone}`);
        }
    });

    await check('version display shows v0.3.0 with the upstream base; app.getVersion() and the saved-chain stamp unchanged', () => {
        const installedJson = JSON.parse(fs.readFileSync(path.join(roots.installedApp, 'resources/app/package.json'), 'utf8'));
        assert.equal(installedJson.version, UPSTREAM_VERSION, 'installed package.json version is the shown upstream base');
        assert.equal(packageJson.version, installedJson.version, 'package.json version');
        assert.equal(count(originalMain, 'Wg.app.getVersion()'), 4, 'installed: About, Release Notes, system information, xf');
        assert.equal(count(mainBundle, 'Wg.app.getVersion()'), 1, 'packaged: xf only');
        for (const unchanged of ['xf=Wg.app.getVersion()', 'returnValue=xf', 'N0.write(I,g,xf)', 'N0.write(n,o,xf)']) {
            assert.equal(count(originalMain, unchanged), 1, `installed: ${unchanged}`);
            assert.equal(count(mainBundle, unchanged), 1, `packaged: ${unchanged}`);
        }
    });

    await check('integrated Python: 3.14.0 or newer kept; a missing or older one throws, nothing deleted or downloaded', async () => {
        const installedRun = (exists, version) => integratedPython(originalMain, 'installed-integrated-python.js', exists, version);
        const packagedRun = (exists, version) => integratedPython(mainBundle, 'packaged-integrated-python.js', exists, version);
        const probe = ['probe', 'C:/package/python/python.exe'];
        assert.deepEqual(await installedRun(true, '3.11.5'), { python: { version: '3.11.5' }, calls: [probe] });
        const replaced = await installedRun(true, '3.14.8');
        assert.deepEqual(replaced.calls.slice(0, 3), [probe, ['rm', 'C:/package//python'], ['download', 'https://example.invalid/python.tar.gz']], 'installed: a 3.14 runtime is deleted and 3.11.5 downloaded');
        for (const version of ['3.14.0', '3.14.8', '3.15.1']) {
            assert.deepEqual(await packagedRun(true, version), { python: { version }, calls: [probe] }, `packaged keeps ${version}`);
        }
        assert.deepEqual(await packagedRun(true, '3.11.5'), { error: REDOWNLOAD_ERROR, calls: [probe] });
        assert.deepEqual(await packagedRun(true, '3.13.9'), { error: REDOWNLOAD_ERROR, calls: [probe] });
        assert.deepEqual(await packagedRun(false, '3.14.8'), { error: REDOWNLOAD_ERROR, calls: [] });
    });

    await check('forbidden update strings: present in installed renderer, absent from packaged', () => {
        for (const forbidden of FORBIDDEN_RENDERER) {
            assert.equal(count(originalRenderer, forbidden), 1, `installed: ${forbidden}`);
            assert.equal(count(renderer, forbidden), 0, `packaged: ${forbidden}`);
        }
    });
    await check('input-drop anchors: each before once in installed, each after once in packaged', () => {
        assert.ok(dropEdits.edits.length > 0);
        for (const edit of dropEdits.edits) {
            assert.equal(count(originalRenderer, edit.before), 1, `installed before: ${edit.label}`);
            assert.equal(count(renderer, edit.after), 1, `packaged after: ${edit.label}`);
        }
    });

    await check('TensorRT types: each before once in installed, each after once in packaged; scope, accent and both colour tokens', () => {
        assert.deepEqual([...new Set(trt.edits.map((edit) => edit.bundle))].sort(), [CSS, MAIN, RENDERER].sort());
        for (const edit of trt.edits) {
            assert.equal(count(bundles.installed[edit.bundle], edit.before), 1, `installed before: ${edit.label}`);
            assert.equal(count(bundles.installed[edit.bundle], edit.after), 0, `installed after: ${edit.label}`);
            assert.equal(count(bundles.packaged[edit.bundle], edit.after), 1, `packaged after: ${edit.label}`);
        }
        for (const text of [renderer, mainBundle]) {
            for (const definition of ['struct TensorRTEngine {', 'enum TrtPrecision { fp32, fp16 }', 'enum TrtShapeMode { fixed, dynamic }', 'def convenientUpscaleTrt(engine: TensorRTEngine, image: Image) {']) {
                assert.equal(count(text, definition), 1, definition);
            }
        }
        assert.equal(count(renderer, '{type:jO("TensorRTEngine"),color:fb("--type-color-tensorrt")}'), 1, 'accent colour');
        const css = bundles.packaged[CSS];
        assert.equal(count(css, '--type-color-tensorrt: #76b900}:root[data-theme=dark]{'), 1, 'light token');
        assert.equal(count(css, '--type-color-tensorrt: #90c41c}'), 1, 'dark token');
        assert.equal(sha256(revertTrt(css, CSS)), sha256(bundles.installed[CSS]), 'stylesheet unchanged outside the tokens');
    });

    await check('TensorRT Clear: each before once in installed, each after once in packaged; menu item and removal clears', () => {
        assert.deepEqual([...new Set(trtClear.edits.map((edit) => edit.bundle))], [RENDERER]);
        for (const edit of trtClear.edits) {
            assert.equal(count(bundles.installed[RENDERER], edit.before), 1, `installed before: ${edit.label}`);
            assert.equal(count(bundles.installed[RENDERER], edit.after), 0, `installed after: ${edit.label}`);
            assert.equal(count(renderer, edit.after), 1, `packaged after: ${edit.label}`);
        }
        assert.equal(count(renderer, 'trtClearBackend.clearNodeCacheIndividual(d).catch(H2.error)},children:"Clear"})'), 1, 'Clear item');
        assert.equal(count(renderer, 'ne.forEach(I2=>{w.clearNodeCacheIndividual(I2).catch(H2.error)})'), 1, 'removal clears');
    });

    await check('renderer unchanged outside the reviewed updater/settings, input-drop and TensorRT edits', () => {
        function normalizeRenderer(text, old) {
            if (!old) {
                text = revertTrt(text, RENDERER);
                for (const edit of [...dropEdits.edits].reverse()) {
                    assert.equal(count(text, edit.after), 1, `Missing/duplicate drop repair: ${edit.label}`);
                    text = text.replace(edit.after, edit.before);
                }
            }
            let normalized = maskRegion(text, old ? ',ln0="chaiNNer-org/chaiNNer"' : ',llr=W.memo(', ',clr=W.memo(', 'header');
            normalized = maskRegion(normalized, ',Yor=W.memo(', ',Qor=W.memo(', 'advanced-settings');
            const flag = old ? 'checkForUpdatesOnStartup:!0' : 'checkForUpdatesOnStartup:!1';
            assert.equal(count(normalized, flag), 1);
            return normalized.replace(flag, 'checkForUpdatesOnStartup:<disabled>');
        }
        assert.equal(sha256(normalizeRenderer(renderer, false)), sha256(normalizeRenderer(originalRenderer, true)));
    });
    await check('main bundle unchanged outside its settings region, the integrated-Python check, the main edits and the TensorRT types', () => {
        const mask = (text) => maskRegion(text, 'const pH=', ',rY=eB.join(', 'settings');
        let reverted = revertTrt(mainBundle, MAIN);
        for (const edit of [{ before: PYTHON_CHECK_UPSTREAM, after: PYTHON_CHECK_PORT, label: 'integrated-Python check' }, ...MAIN_EDITS].reverse()) {
            assert.equal(count(reverted, edit.after), 1, `Missing/duplicate main edit: ${edit.label}`);
            reverted = reverted.replace(edit.after, () => edit.before);
        }
        assert.equal(sha256(mask(reverted)), sha256(mask(originalMain)));
    });

    await check('installed bundles unchanged by this run', () => {
        for (const relative of [RENDERER, MAIN, CSS]) {
            assert.equal(sha256(fs.readFileSync(path.join(roots.installedApp, relative))), INSTALLED_SHA256[relative], relative);
        }
    });
});
