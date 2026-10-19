import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, expect, test } from 'vitest';
import { writeFileAtomic, writeFileAtomicSync } from '../../src/main/util';

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
