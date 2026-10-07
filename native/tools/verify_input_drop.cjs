/* Execute the shipped file-input, node, chain-file and canvas-file drop
 * handlers with controlled drag payloads, against explicit expectations and
 * the pinned installed baseline. Handler semantics only; native Explorer/OLE
 * delivery is a separate gate. Writes nothing.
 * Usage: node verify_input_drop.cjs [--manifest <file>] [--installed-app <dir>] [--package <dir>]
 */
'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
const vm = require('node:vm');
const { spawnSync } = require('node:child_process');
const { RENDERER, loadBundles, region, runChecks } = require('./bundle_verify_common.cjs');

const extensionlessFolder = 'D:\\Datasets\\HR\\0';
const onlyOne = 'Only one file is accepted by Image.';

function section(text, left, right) {
    return region(text, left, right).text.slice(left.length);
}
function run(code, context) {
    return vm.runInNewContext(code, context);
}

function transfer(name = 'image.PNG', { directory = false, count = 1, filesType = true, nativePath = name, textFirst = false } = {}) {
    const files = Array.from({ length: count }, () => ({ path: nativePath, name }));
    const items = files.map(() => ({ kind: 'file', webkitGetAsEntry: () => ({ isDirectory: directory, isFile: !directory }) }));
    if (textFirst) items.unshift({ kind: 'string', webkitGetAsEntry: () => null });
    return { types: filesType ? (textFirst ? ['text/plain', 'Files'] : ['Files']) : ['text/plain'], files, items,
        getData: () => '', dropEffect: 'unset' };
}
function resultContext(data, locked, connected) {
    const out = { values: [], toasts: [], prevented: 0, stopped: 0, effects: [] };
    const input = { id: 7, kind: 'file', label: 'Image', filetypes: ['.png'] };
    const nodeState = { isLocked: locked, connectedInputs: new Set(connected ? [7] : []) };
    const event = { dataTransfer: data, preventDefault: () => out.prevented++, stopPropagation: () => out.stopped++ };
    const context = { w81: path.win32, zc: path.win32,
        d: locked, f: connected, a: (value) => out.values.push(value), H: (value) => out.toasts.push({ ...value }), b: ['.png'], w: 'Image',
        o: nodeState, Z: input, s: (_id, value) => out.values.push(value),
    };
    return { out, event, context };
}
function handler(bundle, kind, data, locked = false, connected = false) {
    const { out, event, context } = resultContext(data, locked, connected);
    let declarations = `const oW1=${section(bundle, 'oW1=', ',TNn=')};`;
    let drop;
    let over;
    if (kind === 'file') {
        declarations += `const A=k=>{k.preventDefault()${section(bundle, '},A=k=>{k.preventDefault()', ',E=S01(p,l,e,f)')};`;
        drop = 'T'; over = 'A';
    } else {
        declarations += `const j=g1=>{g1.preventDefault()${section(bundle, ',j=g1=>{g1.preventDefault()', ',{reload:J,isLive:X}')};`;
        context.f = (value) => out.toasts.push({ ...value });
        drop = 'K'; over = 'j';
    }
    context.event = event;
    run(`${declarations}\n${over}(event);${drop}(event);`, context);
    out.dropEffect = data.dropEffect;
    return JSON.parse(JSON.stringify(out));
}
function openChainFile(bundle, data) {
    const calls = [];
    const ipc = { invoke: (...args) => { calls.push(args); return Promise.resolve({}); }, sendTo: () => {} };
    const context = { Jn: ipc, H2: { error: () => {} }, data };
    const opened = run(`const openChainnerFileProcessor=${section(bundle, 'NNn=', ',DNn=')};\nopenChainnerFileProcessor(data);`, context);
    return { opened, calls };
}
function openCanvasFile(bundle, data) {
    const created = [];
    const context = { w81: path.win32, data,
        options: { schemata: { schemata: [{ schemaId: 'image-load', inputs: [{ id: 0, kind: 'file', primaryInput: true, filetypes: ['.png'] }] }] },
            getNodePosition: () => ({ x: 1, y: 2 }), createNode: (node) => created.push(node) } };
    const code = `const oW1=${section(bundle, 'oW1=', ',TNn=')};const openFileProcessor=${section(bundle, 'DNn=', ',RNn=')};`;
    const opened = run(`${code}\nopenFileProcessor(data,options);`, context);
    return { opened, created };
}

// Expected shipped behavior of the file-input and node handlers (dragover then drop).
const scenarios = [
    { label: 'ordinary-file', config: {}, values: ['image.PNG'] },
    { label: 'uppercase-extension', config: { name: 'IMAGE.PNG' }, values: ['IMAGE.PNG'] },
    { label: 'spaces-unicode', config: { name: 'F:\\New folder\\é image.PNG' }, values: ['F:\\New folder\\é image.PNG'] },
    { label: 'locked', config: { locked: true }, dropEffect: 'none' },
    { label: 'connected', config: { connected: true }, dropEffect: 'none' },
    { label: 'locked-connected', config: { locked: true, connected: true }, dropEffect: 'none' },
    { label: 'multiple', config: { count: 2 }, toast: onlyOne },
    { label: 'empty', config: { count: 0 }, toast: onlyOne },
    { label: 'unsupported-file', config: { name: 'image.txt' }, toast: 'Image does not accept .txt files.' },
    { label: 'folder-named-png', config: { name: 'folder.png', directory: true }, toast: 'Image does not accept .png files.' },
    { label: 'text-first-file', config: { textFirst: true }, values: ['image.PNG'] },
    { label: 'text-first-folder-named-png', config: { name: 'folder.png', directory: true, textFirst: true }, toast: 'Image does not accept .png files.' },
    // Upstream message format with an empty extension (two spaces).
    { label: 'extensionless-folder', config: { name: extensionlessFolder, directory: true }, toast: 'Image does not accept  files.' },
    { label: 'non-files', config: { filesType: false }, dropEffect: 'unset', stopped: 0 },
];
function expected({ values = [], toast, dropEffect = 'copy', stopped = 2 }) {
    return { values, toasts: toast ? [{ status: 'error', description: toast }] : [], prevented: 2, stopped, effects: [], dropEffect };
}

runChecks('verify_input_drop', async (check) => {
    const bundles = await loadBundles(check, [RENDERER]);
    if (!bundles) return;
    const installed = bundles.installed[RENDERER];
    const packaged = bundles.packaged[RENDERER];

    await check('packaged renderer module syntax', () => {
        const syntax = spawnSync(process.execPath, ['--input-type=module', '--check'], { input: packaged, encoding: 'utf8', windowsHide: true });
        if (syntax.error) throw syntax.error;
        assert.equal(syntax.status, 0, syntax.stderr);
    });
    for (const kind of ['file', 'node']) {
        for (const scenario of scenarios) {
            const { config } = scenario;
            await check(`${kind}/${scenario.label}`, () => {
                const data = transfer(config.name || 'image.PNG', config);
                assert.deepEqual(handler(packaged, kind, data, config.locked, config.connected), expected(scenario));
            });
        }
        await check(`${kind}/baseline-locked-bypass-reproduced`, () => assert.equal(handler(installed, kind, transfer(), true).values.length, 1));
        await check(`${kind}/baseline-connected-bypass-reproduced`, () => assert.equal(handler(installed, kind, transfer(), false, true).values.length, 1));
        await check(`${kind}/baseline-folder-extension-misroute-reproduced`, () => assert.equal(handler(installed, kind, transfer('folder.png', { directory: true })).values.length, 1));
    }
    for (const [label, name, directory, accepted, textFirst = false] of [
        ['chain-file', 'sample.chn', false, true], ['uppercase-chain', 'sample.CHN', false, true],
        ['folder-named-chain', 'folder.chn', true, false], ['ordinary-folder', extensionlessFolder, true, false],
        ['image-file', 'image.png', false, false],
        ['text-first-chain-file', 'sample.chn', false, true, true], ['text-first-folder-named-chain', 'folder.chn', true, false, true],
    ]) {
        await check(`chain/${label}`, () => {
            const { opened, calls } = openChainFile(packaged, transfer(name, { directory, textFirst }));
            assert.equal(opened, accepted);
            assert.equal(calls.length, accepted ? 1 : 0);
            if (accepted) assert.equal(calls[0][1], name);
        });
    }
    await check('chain/baseline-folder-named-chain-misroute-reproduced', () => {
        const { opened, calls } = openChainFile(installed, transfer('folder.chn', { directory: true }));
        assert.equal(opened, true);
        assert.equal(calls.length, 1);
    });
    for (const [label, data, accepted] of [
        ['image', transfer('A.PNG'), true], ['folder-named-image', transfer('A.PNG', { directory: true }), false],
        ['directory', transfer(extensionlessFolder, { directory: true }), false], ['multi', transfer('A.PNG', { count: 2 }), false],
        ['text-first-image', transfer('A.PNG', { textFirst: true }), true], ['text-first-folder-named-image', transfer('A.PNG', { directory: true, textFirst: true }), false],
    ]) {
        await check(`canvas-file/${label}`, () => {
            const { opened, created } = openCanvasFile(packaged, data);
            assert.equal(opened, accepted);
            assert.equal(created.length, accepted ? 1 : 0);
        });
    }
    await check('canvas-file/baseline-folder-named-image-misroute-reproduced', () => {
        const { opened, created } = openCanvasFile(installed, transfer('A.PNG', { directory: true }));
        assert.equal(opened, true);
        assert.equal(created.length, 1);
    });
});
