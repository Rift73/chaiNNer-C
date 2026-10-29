import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { getSingleFileWithExtension } from '../../src/renderer/helpers/dataTransfer';

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

test('a file input of any type ("*") takes any one dropped file', () => {
    expect(getSingleFileWithExtension(drop('C:\\models\\x.onnx'), ['*'])).toBe(
        'C:\\models\\x.onnx'
    );
    expect(getSingleFileWithExtension(drop('C:\\shaders\\FSRCNNX'), ['*'])).toBe(
        'C:\\shaders\\FSRCNNX'
    );
    expect(getSingleFileWithExtension(drop('C:\\a.onnx', 'C:\\b.onnx'), ['*'])).toBeUndefined();
});
