import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { isInFolder, writeFileAtomic, writeFileAtomicSync } from '../../src/main/util';

// what reaches the disk, in order: a flush of an open file, a rename
const { steps } = vi.hoisted(() => ({ steps: [] as string[] }));
vi.mock('fs', async (importOriginal) => {
    const actual = await importOriginal<typeof import('fs')>();
    return {
        ...actual,
        fsyncSync: (fd: number) => {
            steps.push('flush');
            actual.fsyncSync(fd);
        },
        renameSync: (from: string, to: string) => {
            steps.push('rename');
            actual.renameSync(from, to);
        },
    };
});
vi.mock('fs/promises', async (importOriginal) => {
    const actual = await importOriginal<typeof import('fs/promises')>();
    const open: typeof actual.open = async (...args) => {
        const handle = await actual.open(...args);
        const sync = handle.sync.bind(handle);
        handle.sync = () => {
            steps.push('flush');
            return sync();
        };
        return handle;
    };
    const rename: typeof actual.rename = (from, to) => {
        steps.push('rename');
        return actual.rename(from, to);
    };
    return { ...actual, default: { ...actual, open, rename }, open, rename };
});

const MIB = 1024 * 1024;

let dir: string;

beforeEach(() => {
    dir = mkdtempSync(path.join(os.tmpdir(), 'chaiNNer-util-test-'));
});

afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
});

test('overlapping atomic writes of one file never share a temporary file', async () => {
    const file = path.join(dir, 'chain.chn');
    // Large enough that a write is still in progress when the next one starts
    const contents = Array.from({ length: 16 }, (_, i) => String(i).repeat(MIB));

    await Promise.all(contents.map((content) => writeFileAtomic(file, content)));

    expect(contents).toContain(readFileSync(file, 'utf-8'));
    expect(readdirSync(dir)).toEqual(['chain.chn']);
});

test('a synchronous write during asynchronous ones leaves one whole file', async () => {
    const file = path.join(dir, 'settings.json');
    const pending = [
        writeFileAtomic(file, 'a'.repeat(MIB)),
        writeFileAtomic(file, 'b'.repeat(MIB)),
    ];
    writeFileAtomicSync(file, 'c'.repeat(MIB));

    await Promise.all(pending);

    expect(['a', 'b', 'c'].map((c) => c.repeat(MIB))).toContain(readFileSync(file, 'utf-8'));
    expect(readdirSync(dir)).toEqual(['settings.json']);
});

test('a file is in a folder only below it, not in a folder whose name starts the same', () => {
    const folder = path.join(dir, 'chaiNNer-C');
    expect(isInFolder(path.join(folder, 'chain.chn'), folder)).toBe(true);
    expect(isInFolder(path.join(folder, 'chains', '..chn'), folder)).toBe(true);
    expect(isInFolder(path.join(`${folder} chains`, 'chain.chn'), folder)).toBe(false);
    expect(isInFolder(path.join(dir, 'chain.chn'), folder)).toBe(false);
});

// a rename can reach the disk before the file's content: a power loss would leave the name empty
test.each<[string, (file: string) => Promise<void>]>([
    ['an atomic write', (file: string) => writeFileAtomic(file, 'chain')],
    [
        'a synchronous atomic write',
        (file: string) => Promise.resolve().then(() => writeFileAtomicSync(file, 'chain')),
    ],
])('%s flushes the new content before the rename', async (_, write) => {
    steps.length = 0;
    const file = path.join(dir, 'chain.chn');
    await write(file);
    expect(steps).toEqual(['flush', 'rename']);
    expect(readFileSync(file, 'utf-8')).toBe('chain');
});
