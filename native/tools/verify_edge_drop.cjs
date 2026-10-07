/* Handler-routing regression for the packaged edge-drop forwarding helper.
 * DOM classes are controlled here; the live Electron hit-test is recorded
 * separately (probe_edge_drop_browser.js). This is not an Explorer/OLE test.
 * Writes nothing.
 * Usage: node verify_edge_drop.cjs [--manifest <file>] [--installed-app <dir>] [--package <dir>]
 */
'use strict';
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { RENDERER, count, loadBundles, region, runChecks } = require('./bundle_verify_common.cjs');

const wiring = 'onDragOverCapture:cnForwardFileDrop,onDropCapture:cnForwardFileDrop';

function scenario(code, config = {}) {
    let handler;
    const out = { prevented: 0, stopped: 0, hitTests: 0, deliveries: 0, receiver: null, canvas: 0 };
    class Element {
        constructor(kind, inside = true) { this.kind = kind; this.inside = inside; }
        closest(selector) { return (selector === '.react-flow__edge' && this.kind === 'edge') || (selector === '.react-flow__node' && this.kind === 'node') ? this : null; }
        dispatchEvent(forwarded) {
            out.deliveries++;
            assert.equal(forwarded.dataTransfer, transfer, 'Must retain native DataTransfer object');
            for (const key of ['clientX', 'clientY', 'screenX', 'screenY', 'ctrlKey', 'shiftKey', 'altKey', 'metaKey', 'button', 'buttons', 'relatedTarget']) assert.equal(forwarded[key], event[key], key);
            assert.equal(forwarded.bubbles, true); assert.equal(forwarded.cancelable, true); assert.equal(forwarded.composed, true);
            assert.equal(forwarded.type, event.type);
            const nested = { ...forwarded, target: this, currentTarget: wrapper, nativeEvent: forwarded,
                preventDefault() {}, stopPropagation() {} };
            handler(nested); // Real capture is reentered by dispatchEvent; must not recurse.
            out.receiver = this === upper ? 'upper' : 'lower';
            if (config.locked || config.connected) transfer.dropEffect = 'none';
            else { transfer.dropEffect = 'copy'; out.accepted = config.type !== 'dragover'; }
            return false;
        }
    }
    class DragEvent { constructor(type, options) { Object.assign(this, options, { type }); } }
    const edge = new Element('edge'), upper = new Element('node', !config.outside), lower = new Element('node');
    const transfer = { types: config.nonFiles ? ['application/chainner/schema'] : ['Files'], dropEffect: 'unset' };
    const wrapper = { contains: (e) => e.inside, ownerDocument: { elementsFromPoint(x, y) {
        out.hitTests++; assert.equal(x, 41); assert.equal(y, 57);
        return config.noNode ? [edge] : config.overlap ? [edge, upper, lower] : [edge, upper];
    } } };
    const event = { type: config.type || 'drop', target: config.nonElement ? {} : config.direct ? upper : edge, currentTarget: wrapper, dataTransfer: transfer,
        clientX: 41, clientY: 57, screenX: 141, screenY: 257, ctrlKey: true, shiftKey: false, altKey: true, metaKey: false, button: 0, buttons: 1, relatedTarget: null, nativeEvent: { composed: true },
        preventDefault() { out.prevented++; }, stopPropagation() { out.stopped++; } };
    const context = { Element, DragEvent, exports: {} };
    vm.runInNewContext(code, context);
    handler = context.exports.forwardFileDropPastEdge;
    handler(event);
    if (!out.stopped) out.canvas++;
    out.effect = transfer.dropEffect;
    return out;
}

runChecks('verify_edge_drop', async (check) => {
    const bundles = await loadBundles(check, [RENDERER]);
    if (!bundles) return;
    const installed = bundles.installed[RENDERER];
    const bundle = bundles.packaged[RENDERER];
    let code;
    if (!await check('packaged helper present exactly once', () => {
        code = region(bundle, 'const cnForwardFileDrop=', ',oW1=').text + ';exports.forwardFileDropPastEdge=cnForwardFileDrop;';
    })) return;

    for (const type of ['drop', 'dragover']) {
        await check(`packaged/${type}/edge-over-unselected-node`, () => {
            const r = scenario(code, { type });
            assert.equal(r.deliveries, 1); assert.equal(r.prevented, 1); assert.equal(r.stopped, 1);
            assert.equal(r.canvas, 0); assert.equal(r.effect, 'copy'); assert.equal(r.accepted, type === 'drop');
        });
        for (const guard of ['locked', 'connected']) await check(`packaged/${type}/${guard}`, () => {
            const r = scenario(code, { type, [guard]: true });
            assert.equal(r.deliveries, 1); assert.equal(r.effect, 'none'); assert.equal(r.accepted, undefined); assert.equal(r.canvas, 0);
        });
    }
    for (const bypass of ['nonFiles', 'direct', 'nonElement', 'noNode', 'outside']) await check(`packaged/${bypass}/unchanged-routing`, () => {
        const r = scenario(code, { [bypass]: true });
        assert.equal(r.deliveries, 0); assert.equal(r.prevented, 0); assert.equal(r.stopped, 0); assert.equal(r.canvas, 1);
    });
    await check('packaged/overlapping-nodes/topmost-only', () => {
        const r = scenario(code, { overlap: true }); assert.equal(r.receiver, 'upper'); assert.equal(r.deliveries, 1);
    });
    await check('packaged capture wiring on the elevated-edge flow wrapper', () => {
        assert.equal(count(bundle, wiring), 1);
        assert.match(bundle, /elevateEdgesOnSelect:!0/);
    });
    await check('installed baseline elevates edges without forwarding (the repaired defect)', () => {
        assert.match(installed, /elevateEdgesOnSelect:!0/);
        assert.equal(count(installed, 'cnForwardFileDrop'), 0);
    });
});
