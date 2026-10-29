import { existsSync } from 'fs';
import { mkdir, mkdtemp, rm, writeFile } from 'fs/promises';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, describe, expect, test } from 'vitest';
import forgeConfig from '../../forge.config';

type PostPackage = (
    config: unknown,
    result: { platform: string; arch: string; outputPaths: string[] }
) => Promise<void>;
const postPackage = forgeConfig.hooks?.postPackage as unknown as PostPackage;

let outputPath: string;

beforeEach(async () => {
    outputPath = await mkdtemp(path.join(os.tmpdir(), 'chaiNNer-package-test-'));
    const backend = path.join(outputPath, 'resources/src');
    await mkdir(path.join(backend, 'texconv'), { recursive: true });
    await writeFile(path.join(backend, 'texconv/texconv.exe'), '');
    await writeFile(path.join(backend, 'texconv/LICENSE'), '');
    await writeFile(path.join(backend, 'server.py'), '');
});

afterEach(async () => {
    await rm(outputPath, { recursive: true, force: true });
});

describe('packaging', () => {
    test('the Linux package has no texconv.exe, a Windows program, and keeps the backend', async () => {
        await postPackage({}, { platform: 'linux', arch: 'x64', outputPaths: [outputPath] });

        expect(existsSync(path.join(outputPath, 'resources/src/texconv'))).toBe(false);
        expect(existsSync(path.join(outputPath, 'resources/src/server.py'))).toBe(true);
    });

    test.each(['win32', 'darwin'])('the %s package keeps texconv.exe', async (platform) => {
        await postPackage({}, { platform, arch: 'x64', outputPaths: [outputPath] });

        expect(existsSync(path.join(outputPath, 'resources/src/texconv/texconv.exe'))).toBe(true);
    });
});
