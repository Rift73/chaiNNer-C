/* Offline regression for the packaged DirectoryInput drop repair. Executes the
 * installed and packaged minified component in a VM with explicit UI/native-
 * path stubs, not Electron. No drop or GUI is simulated as a live Windows
 * result. Writes nothing.
 * Usage: node verify_directory_drop.cjs [--manifest <file>] [--installed-app <dir>] [--package <dir>]
 */
'use strict';

const assert = require('node:assert/strict');
const vm = require('node:vm');
const { RENDERER, loadBundles, region, runChecks } = require('./bundle_verify_common.cjs');

const defaultFolder = 'C:\\fixtures\\folder';
const onlyOne = 'Only one directory is accepted by Directory.';
const notAFile = 'Drop a directory onto Directory, not a file.';

function directoryInput(bundle) {
    return region(bundle, ',bir=W.memo(', ',eJ=[0,1,2],').text.slice(',bir='.length);
}

function transfer(options = {}) {
    const file = { nativePath: Object.hasOwn(options, 'rawPath') ? options.rawPath : options.path ?? defaultFolder, pathError: options.pathError };
    Object.defineProperty(file, 'path', { get() {
        if (file.pathError) throw new Error('Native path unavailable');
        return file.nativePath;
    } });
    const item = { kind: options.kind ?? 'file', webkitGetAsEntry() {
        assert.equal(this, item, 'Entry method keeps its DOM receiver');
        if (options.entryError) throw new Error('Entry metadata unavailable');
        return options.nullEntry ? null : { isDirectory: options.directory !== false };
    } };
    if (options.missingMethod) delete item.webkitGetAsEntry;
    return {
        types: options.types ?? ['Files'],
        files: options.count === 0 ? [] : options.count === 2 ? [file, { ...file }] : [file],
        items: options.noItems ? [] : options.extraFile ? [item, item] : [item],
        dropEffect: 'none',
        getData() { throw new Error('Text must never be parsed'); },
    };
}

function element(type, props = {}) { return { type, props }; }
function first(tree, type) {
    if (!tree || typeof tree !== 'object') return undefined;
    if (tree.type === type) return tree;
    for (const value of Object.values(tree)) {
        const found = Array.isArray(value) ? value.map((x) => first(x, type)).find(Boolean) : first(value, type);
        if (found) return found;
    }
    return undefined;
}
function canonical(value) {
    if (typeof value === 'function') return '[function]';
    if (Array.isArray(value)) return Array.from(value, canonical);
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value)
        .filter(([key]) => !['onDragOver', 'onDrop'].includes(key))
        .map(([key, item]) => [key, canonical(item)]));
    return value;
}
function render(component, changes = {}) {
    const events = [];
    let selection = { canceled: false, filePaths: ['C:\\selected'] };
    const noop = () => {};
    const sandbox = {
        W: { memo: (f) => f }, _: { jsx: element, jsxs: element }, Y8: () => ({ t: (_, text) => text }),
        We: () => ({ sendToast: (value) => events.push(['toast', value.status, value.description]) }), Cm: {},
        X20: () => ({ lastDirectory: 'C:\\previous', setLastDirectory: (value) => events.push(['last', value]) }),
        S01: noop, eg: () => ({ onContextMenu: noop }),
        Jn: { invoke: async (...args) => { events.push(['ipc', ...args]); return selection; } },
        H2: { error: (error) => events.push(['log', String(error)]) },
        Cir: (type) => type?.kind === 'directory' && type.fields.path?.kind === 'literal' ? type.fields.path.value : undefined,
        h7: 'MenuList', Or: 'MenuItem', Hf: 'MenuDivider', LZ1: 'BsFolderPlus', Rq1: 'MdFolder', iH: 'MdContentCopy', Nv: 'CloseIcon',
        xd1: 'AutoLabel', i5: 'Tooltip', Jb: 'InputGroup', WL: 'InputLeftElement', ma: 'Icon', wm: 'Input',
    };
    new vm.Script(`globalThis.component = ${component};`, { filename: 'DirectoryInput.js' }).runInContext(vm.createContext(sandbox), { timeout: 2000 });
    const props = { value: undefined, setValue: (value) => events.push(['value', value]), resetValue: noop,
        isLocked: false, isConnected: false, input: { kind: 'directory', label: 'Directory' },
        inputKey: 'directory', inputType: { kind: 'directory', fields: { path: { kind: 'literal', value: 'C:\\connected' } } },
        nodeId: 'node', ...changes };
    const tree = sandbox.component(props);
    return { tree, events, group: first(tree, 'InputGroup'), input: first(tree, 'Input'),
        setSelection: (value) => { selection = value; } };
}
function event(dataTransfer) {
    return { dataTransfer, prevented: 0, stopped: 0,
        preventDefault() { this.prevented += 1; }, stopPropagation() { this.stopped += 1; } };
}
function dragThenDrop(component, props, options) {
    const view = render(component, props);
    const over = event(transfer(options));
    const drop = event(transfer(options));
    view.group.props.onDragOver(over);
    view.group.props.onDrop(drop);
    return { events: view.events, over: [over.prevented, over.stopped, over.dataTransfer.dropEffect], drop: [drop.prevented, drop.stopped] };
}

const accepted = [
    ['folder', {}], ['plain folder', { path: 'C:\\folder' }], ['dotted folder', { path: 'C:\\folder.v1' }],
    ['image-suffix folder', { path: 'C:\\folder.png' }], ['picture-named folder', { path: 'C:\\picture.png' }],
    ['chain-suffix folder', { path: 'C:\\folder.chn' }], ['Unicode folder', { path: 'C:\\資料 空白' }],
    ['Unicode nested folder', { path: 'C:\\空 白\\folder' }], ['UNC folder', { path: '\\\\server\\share\\folder' }],
];
const rejected = [
    ['regular file', { directory: false, path: 'C:\\image.png' }],
    ['extensionless file', { directory: false, path: 'C:\\README' }],
    ['zero files', { count: 0 }], ['multiple files', { count: 2 }],
    ['empty path', { path: '' }], ['native path error', { pathError: true }],
    ['null entry', { nullEntry: true }], ['entry error', { entryError: true }],
    ['missing entry API', { missingMethod: true }], ['no items', { noItems: true }],
    ['string item', { kind: 'string' }], ['ambiguous file items', { extraFile: true }],
    ...[undefined, null, 0, {}].map((rawPath) => [`non-string native path ${String(rawPath)}`, { rawPath }]),
];
const ignored = [['plain text drag', { types: ['text/plain'] }], ['empty drag types', { types: [] }]];

function expected(kind, options, { isLocked, isConnected }) {
    if (kind === 'ignored') return { events: [], over: [0, 0, 'none'], drop: [0, 0] };
    if (isLocked || isConnected) return { events: [], over: [1, 1, 'none'], drop: [1, 1] };
    const events = kind === 'accepted'
        ? [['value', options.path ?? defaultFolder], ['last', options.path ?? defaultFolder]]
        : [['toast', 'error', options.count === 0 || options.count === 2 ? onlyOne : notAFile]];
    return { events, over: [1, 1, 'copy'], drop: [1, 1] };
}

runChecks('verify_directory_drop', async (check) => {
    const bundles = await loadBundles(check, [RENDERER]);
    if (!bundles) return;
    let installed;
    let packaged;
    if (!await check('DirectoryInput component present exactly once in both bundles', () => {
        installed = directoryInput(bundles.installed[RENDERER]);
        packaged = directoryInput(bundles.packaged[RENDERER]);
    })) return;

    await check('installed component reproduces missing directory-drop handlers', () => {
        const old = render(installed);
        assert.equal(old.group.props.onDrop, undefined);
        assert.equal(old.group.props.onDragOver, undefined);
    });
    for (const isLocked of [false, true]) for (const isConnected of [false, true]) {
        const props = { isLocked, isConnected };
        const suffix = `locked=${isLocked} connected=${isConnected}`;
        await check(`visible structure unchanged from installed ${suffix}`, () => {
            assert.deepEqual(canonical(render(packaged, props).tree), canonical(render(installed, props).tree));
        });
        for (const [kind, cases] of [['accepted', accepted], ['rejected', rejected], ['ignored', ignored]]) {
            for (const [name, options] of cases) await check(`${kind} ${name} ${suffix}`, () => {
                assert.deepEqual(dragThenDrop(packaged, props, options), expected(kind, options, props));
            });
        }
    }
    await check('repeat successful drop preserves explicit setter order', () => {
        const view = render(packaged);
        for (let i = 0; i < 2; i += 1) view.group.props.onDrop(event(transfer()));
        assert.deepEqual(view.events, [0, 1].flatMap(() => [['value', defaultFolder], ['last', defaultFolder]]));
    });
    for (const selection of [{ canceled: false, filePaths: ['C:\\selected'] }, { canceled: true, filePaths: ['C:\\ignored'] }, { canceled: false, filePaths: [] }]) {
        await check(`directory picker unchanged from installed ${JSON.stringify(selection)}`, async () => {
            const old = render(installed);
            const current = render(packaged);
            old.setSelection(selection); current.setSelection(selection);
            await old.input.props.onClick(); await current.input.props.onClick();
            assert.ok(old.events.length > 0);
            assert.deepEqual(current.events, old.events);
        });
    }
});
