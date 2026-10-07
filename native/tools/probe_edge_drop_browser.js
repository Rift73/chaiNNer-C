/* Run in the portable test app's DevTools console. Uses a synthetic OS payload
 * and actual DOM hit-testing/React handlers, never executes the image chain.
 * Restores the input value afterwards. Not proof of Windows OLE delivery. */
(async () => {
    const input = document.querySelector('.react-flow__node-regularNode input[readonly]');
    const node = input.closest('.react-flow__node');
    const original = input.value;
    const selection = () =>
        [...document.querySelectorAll('.react-flow__node')].map((e) => [
            e.dataset.id,
            e.classList.contains('selected'),
        ]);
    const beforeSelection = JSON.stringify(selection());
    const rect = input.getBoundingClientRect();
    const x = rect.x + rect.width / 2;
    const y = rect.y + rect.height / 2;
    const hit = document.elementFromPoint(x, y);
    if (node.classList.contains('selected') || !hit.closest('.react-flow__edge')) {
        throw Error('Fixture must have an elevated edge covering the unselected input');
    }
    const fixture = '<repository root>\\native\\reports\\folder-drop\\output'; // set to this checkout
    if (fixture.startsWith('<repository root>')) {
        throw Error("Set fixture to this checkout's native\\reports\\folder-drop\\output first");
    }
    let delivered = 0;
    const observe = () => {
        delivered += 1;
    };
    input.addEventListener('drop', observe);
    const fire = (folder) => {
        const transfer = new DataTransfer();
        const file = new File([], 'folder');
        Object.defineProperty(file, 'path', { value: folder });
        transfer.items.add(file);
        // Chromium rewraps synthetic File/Item entries when the native lists are
        // read. Supply stable controlled metadata on this test DataTransfer.
        Object.defineProperty(transfer, 'files', { value: [file] });
        Object.defineProperty(transfer, 'items', {
            value: [{ kind: 'file', webkitGetAsEntry: () => ({ isDirectory: true }) }],
        });
        ['dragover', 'drop'].forEach((type) => {
            document.elementFromPoint(x, y).dispatchEvent(
                new DragEvent(type, {
                    bubbles: true,
                    cancelable: true,
                    dataTransfer: transfer,
                    clientX: x,
                    clientY: y,
                })
            );
        });
    };
    let received;
    try {
        fire(fixture);
        await new Promise((resolve) => {
            setTimeout(resolve, 100);
        });
        received = input.value;
    } finally {
        fire(original);
        await new Promise((resolve) => {
            setTimeout(resolve, 100);
        });
        input.removeEventListener('drop', observe);
    }
    const result = {
        kind: 'Live Electron DOM hit-test and synthetic DataTransfer; not native Explorer/OLE',
        hitTag: hit.tagName,
        targetInitiallyUnselected: true,
        accepted: received === fixture,
        restored: input.value === original,
        delivered,
        selectionUnchanged: JSON.stringify(selection()) === beforeSelection,
        canvasWarning: document.body.innerText.includes('Unable to transfer dragged item(s).'),
    };
    console.log(`EDGE_DROP_PROBE ${JSON.stringify(result)}`);
    if (
        !result.accepted ||
        !result.restored ||
        result.delivered !== 2 ||
        !result.selectionUnchanged ||
        result.canvasWarning
    ) {
        throw Error('Edge drop probe failed');
    }
    return result;
})();
