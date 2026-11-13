import { SourceDocument, evaluate, parseExpression } from '@chainner/navi';
import { expect, test } from 'vitest';
import { getChainnerScope } from '../../src/common/types/chainner-scope';

const split = (filePath: string) =>
    evaluate(
        parseExpression(new SourceDocument(`splitFilePath(${JSON.stringify(filePath)})`, 'test')),
        getChainnerScope()
    ).toString();
const parts = (dir: string, basename: string, ext: string) =>
    `SplitFilePath { dir: Directory { path: ${JSON.stringify(dir)} }, basename: ${JSON.stringify(
        basename
    )}, ext: ${JSON.stringify(ext)} }`;

// what the backend's split_file_path (backend/src/nodes/utils/utils.py) gives: os.path.splitext,
// then os.path.split
test.each([
    ['/a/b/file.png', '/a/b', 'file', '.png'],
    ['/a/b/.bashrc', '/a/b', '.bashrc', ''],
    ['/a/b/', '/a/b', '', ''],
    ['/a/b//', '/a/b', '', ''],
    ['/', '/', '', ''],
])('%s splits into %s, %s and %s', (filePath, dir, basename, ext) => {
    expect(split(filePath)).toBe(parts(dir, basename, ext));
});

test.runIf(process.platform === 'win32')('a Windows path ending in a separator has no file', () => {
    expect(split('C:\\x\\')).toBe(parts('C:\\x', '', ''));
    expect(split('C:/x/')).toBe(parts('C:/x', '', ''));
    expect(split('C:\\')).toBe(parts('C:\\', '', ''));
});
