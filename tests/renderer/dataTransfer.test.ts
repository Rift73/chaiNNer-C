import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { SchemaId } from '../../src/common/common-types';
import { SchemaMap } from '../../src/common/SchemaMap';
import {
    dataTransferProcessors,
    getSingleFileWithExtension,
} from '../../src/renderer/helpers/dataTransfer';

const stubs = vi.hoisted(() => ({ invoke: vi.fn() }));
vi.mock('../../src/renderer/safeIpc', () => ({ ipcRenderer: { invoke: stubs.invoke } }));

const drop = (...paths: string[]): DataTransfer => {
    const files = paths.map((path) => ({ name: path } as unknown as File));
    return {
        files,
        items: files.map(() => ({
            kind: 'file',
            webkitGetAsEntry: () => ({ isDirectory: false }),
        })),
    } as unknown as DataTransfer;
};

beforeEach(() => {
    vi.stubGlobal('window', {
        electronFileUtils: { getPathForFile: (file: File) => file.name },
    });
});

afterEach(() => {
    vi.unstubAllGlobals();
});

test('a file input takes a dropped file with one of its extensions', () => {
    expect(getSingleFileWithExtension(drop('C:\\clips\\a.MKV'), ['.mp4', '.mkv'])).toBe(
        'C:\\clips\\a.MKV'
    );
    expect(getSingleFileWithExtension(drop('C:\\models\\x.onnx'), ['.mkv'])).toBeUndefined();
    expect(getSingleFileWithExtension(drop('C:\\a.mkv', 'C:\\b.mkv'), ['.mkv'])).toBeUndefined();
});

test('a dropped file opens as a chain only if its extension says so', () => {
    const loadImage = {
        schemaId: 'test:drop:load' as SchemaId,
        inputs: [{ id: 0, kind: 'file', primaryInput: true, filetypes: ['.png'] }],
        kind: 'regularNode',
    } as unknown as SchemaMap['schemata'][number];
    const createNode = vi.fn();
    const dropFile = (path: string) => {
        stubs.invoke.mockReset().mockResolvedValue(undefined);
        createNode.mockClear();
        const transfer = Object.assign(drop(path), { getData: () => '' });
        for (const processor of dataTransferProcessors) {
            const options = {
                createNode,
                getNodePosition: () => ({ x: 0, y: 0 }),
                schemata: new SchemaMap([loadImage]),
                changeNodes: vi.fn(),
                changeEdges: vi.fn(),
            };
            if (processor(transfer, options)) return;
        }
    };

    dropFile('C:\\chains\\a.CHN');
    expect(stubs.invoke).toHaveBeenCalledWith('open-save-file', 'C:\\chains\\a.CHN');

    for (const image of ['C:\\work.chn\\a.png', 'C:\\a.chn.png', 'C:\\a.chnx.png']) {
        dropFile(image);
        expect(stubs.invoke).not.toHaveBeenCalled();
        expect(createNode).toHaveBeenCalledOnce();
    }
});

test('a file input of any type ("*") takes any one dropped file', () => {
    expect(getSingleFileWithExtension(drop('C:\\models\\x.onnx'), ['*'])).toBe(
        'C:\\models\\x.onnx'
    );
    expect(getSingleFileWithExtension(drop('C:\\shaders\\FSRCNNX'), ['*'])).toBe(
        'C:\\shaders\\FSRCNNX'
    );
    expect(getSingleFileWithExtension(drop('C:\\a.onnx', 'C:\\b.onnx'), ['*'])).toBeUndefined();
});
