import { spawnSync } from 'child_process';
import { existsSync, readFileSync } from 'fs';
import { chmod, copyFile, mkdir, mkdtemp, rm, symlink, writeFile } from 'fs/promises';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, describe, expect, test } from 'vitest';

const launcher = path.join(__dirname, '../../src/public/linux/chaiNNer.sh');
const libraryTable = path.join(__dirname, '../../backend/src/linux_libraries.txt');

let appDir: string;
let toolsDir: string;

const writeScript = async (file: string, body: string) => {
    await writeFile(file, `#!/bin/sh\n${body}\n`);
    await chmod(file, 0o755);
};

beforeEach(async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'chaiNNer-launcher-test-'));
    appDir = path.join(root, 'app');
    toolsDir = path.join(root, 'tools');
    await mkdir(appDir);
    await mkdir(toolsDir);
    await copyFile(launcher, path.join(appDir, 'chaiNNer.sh'));
    await chmod(path.join(appDir, 'chaiNNer.sh'), 0o755);
    // the program prints its arguments, one per line, and ends with the code in STUB_EXIT, if any
    await writeScript(path.join(appDir, 'chainner'), 'printf "%s\n" "$@"\nexit $STUB_EXIT');
    // the launcher's tools, and nothing else, so that a unshare or xvfb-run of the machine is not found
    await Promise.all(
        ['dirname', 'readlink', 'cat', 'stat'].map((tool) => {
            const found = ['/usr/bin', '/bin'].map((dir) => path.join(dir, tool)).find(existsSync);
            return symlink(found!, path.join(toolsDir, tool));
        })
    );
});

afterEach(async () => {
    await rm(path.dirname(appDir), { recursive: true, force: true });
});

const launch = (args: string[], env: Record<string, string> = {}) => {
    const result = spawnSync(path.join(appDir, 'chaiNNer.sh'), args, {
        env: { PATH: toolsDir, ...env },
        encoding: 'utf-8',
    });
    return {
        code: result.status,
        lines: result.stdout.split('\n').slice(0, -1),
        stderr: result.stderr,
    };
};

describe.skipIf(process.platform !== 'linux')('the Linux launcher', () => {
    describe('without a user namespace', () => {
        beforeEach(async () => {
            await writeScript(path.join(toolsDir, 'unshare'), 'exit 1');
        });

        test('starts without the sandbox and says so', () => {
            const { lines, stderr } = launch(['chain.chn']);

            expect(lines).toEqual(['--no-sandbox', 'chain.chn']);
            expect(stderr).toMatch(/^chaiNNer-C: starting without the Chromium sandbox/);
        });

        test('keeps the sandbox for CHAINNER_SANDBOX=1', () => {
            const { lines, stderr } = launch(['chain.chn'], { CHAINNER_SANDBOX: '1' });

            expect(lines).toEqual(['chain.chn']);
            expect(stderr).toBe('');
        });

        test('passes --no-sandbox once when it is among the arguments', () => {
            const { lines, stderr } = launch(['--no-sandbox', 'chain.chn']);

            expect(lines).toEqual(['--no-sandbox', 'chain.chn']);
            expect(stderr).toBe('');
        });
    });

    describe('with a user namespace', () => {
        beforeEach(async () => {
            await writeScript(path.join(toolsDir, 'unshare'), 'exit 0');
        });

        test('keeps the sandbox for a user that is not root', async () => {
            await writeScript(path.join(toolsDir, 'id'), 'echo 1000');

            const { lines, stderr } = launch(['chain.chn']);

            expect(lines).toEqual(['chain.chn']);
            expect(stderr).toBe('');
        });

        test('starts without the sandbox as root, and says so', async () => {
            await writeScript(path.join(toolsDir, 'id'), 'echo 0');

            const { lines, stderr } = launch(['chain.chn']);

            expect(lines).toEqual(['--no-sandbox', 'chain.chn']);
            expect(stderr).toMatch(/^chaiNNer-C: starting without the Chromium sandbox/);
        });

        test('starts without the sandbox as root when running a chain', async () => {
            await writeScript(path.join(toolsDir, 'id'), 'echo 0');

            const { lines } = launch(['run', 'chain.chn'], { DISPLAY: 'x' });

            expect(lines).toEqual(['--no-sandbox', 'run', 'chain.chn']);
        });

        test('keeps the sandbox as root for CHAINNER_SANDBOX=1', async () => {
            await writeScript(path.join(toolsDir, 'id'), 'echo 0');

            const { lines, stderr } = launch(['chain.chn'], { CHAINNER_SANDBOX: '1' });

            expect(lines).toEqual(['chain.chn']);
            expect(stderr).toBe('');
        });
    });

    test('CHAINNER_SANDBOX=0 starts without the sandbox, and says nothing', async () => {
        await writeScript(path.join(toolsDir, 'unshare'), 'exit 0');

        const { lines, stderr } = launch(['chain.chn'], { CHAINNER_SANDBOX: '0' });

        expect(lines).toEqual(['--no-sandbox', 'chain.chn']);
        expect(stderr).toBe('');
    });

    test('ends with the program, and with its exit code', () => {
        expect(launch(['chain.chn'], { CHAINNER_SANDBOX: '1', STUB_EXIT: '7' }).code).toBe(7);
    });

    test('runs from a symbolic link to the launcher', async () => {
        const link = path.join(toolsDir, 'start');
        await symlink(path.join(appDir, 'chaiNNer.sh'), link);

        const result = spawnSync(link, ['chain.chn'], {
            env: { PATH: toolsDir, CHAINNER_SANDBOX: '1' },
            encoding: 'utf-8',
        });

        expect(result.stdout).toBe('chain.chn\n');
    });

    describe('running a chain', () => {
        const env = { CHAINNER_SANDBOX: '1' };

        test('without a display uses xvfb-run when it is installed', async () => {
            await writeScript(path.join(toolsDir, 'xvfb-run'), 'printf "%s\n" xvfb "$@"');

            const { lines } = launch(['run', 'chain.chn'], env);

            expect(lines).toEqual([
                'xvfb',
                '-a',
                path.join(appDir, 'chainner'),
                'run',
                'chain.chn',
            ]);
        });

        test.each(['DISPLAY', 'WAYLAND_DISPLAY'])(
            'with %s starts the program itself',
            async (name) => {
                await writeScript(path.join(toolsDir, 'xvfb-run'), 'printf "%s\n" xvfb "$@"');

                const { lines, stderr } = launch(['run', 'chain.chn'], { ...env, [name]: 'x' });

                expect(lines).toEqual(['run', 'chain.chn']);
                expect(stderr).toBe('');
            }
        );

        test('opening the editor needs no virtual display', () => {
            const { lines, stderr } = launch(['chain.chn'], env);

            expect(lines).toEqual(['chain.chn']);
            expect(stderr).toBe('');
        });
    });
});

describe.skipIf(process.platform !== 'linux')(
    "the launcher's check of the system libraries",
    () => {
        const env = { CHAINNER_SANDBOX: '1' };
        const libraries = readFileSync(libraryTable, 'utf-8')
            .split('\n')
            .filter((line) => line !== '' && !line.startsWith('#'))
            .map((line) => line.split(/\s+/)[0])
            .filter((file) => file.includes('.so'));

        // an ldconfig that lists every library of the table but the missing ones
        const provideLibraries = (missing: string[] = [], abi = '(libc6,x86-64)') => {
            const lines = libraries
                .filter((library) => !missing.includes(library))
                .map((library) => `'\t${library} ${abi} => /usr/lib/${library}'`);
            return writeScript(
                path.join(toolsDir, 'ldconfig'),
                `printf '%s\\n' ${lines.join(' ')}`
            );
        };

        beforeEach(async () => {
            await mkdir(path.join(appDir, 'resources/src'), { recursive: true });
            await copyFile(libraryTable, path.join(appDir, 'resources/src/linux_libraries.txt'));
        });

        test('starts the program when the system has every library', async () => {
            await provideLibraries();

            const { code, lines, stderr } = launch(['chain.chn'], env);

            expect(code).toBe(0);
            expect(lines).toEqual(['chain.chn']);
            expect(stderr).toBe('');
        });

        test.each(['libGL.so.1', 'libgtk-3.so.0'])(
            'ends the launch with the install command when %s is missing',
            async (library) => {
                await provideLibraries([library]);

                const { code, lines, stderr } = launch(['chain.chn'], env);

                expect(code).toBe(1);
                expect(lines).toEqual([]);
                expect(stderr).toContain(
                    `chaiNNer-C cannot start: this system lacks ${library}.\n`
                );
                expect(stderr).toMatch(
                    /\nInstall them with:\n\n {2}(sudo )?(apt-get install|dnf install|pacman -S --needed|zypper install) \S+\n$|\nInstall the packages of this distribution that provide them\.\n$/
                );
            }
        );

        test('names every library that is missing, and counts only 64-bit ones', async () => {
            await provideLibraries([], '(libc6)');

            const { code, stderr } = launch(['chain.chn'], env);

            expect(code).toBe(1);
            expect(stderr).toContain(`this system lacks ${libraries.join(' ')}.\n`);
        });

        test('finds a library in a directory of LD_LIBRARY_PATH', async () => {
            await provideLibraries(['libGL.so.1']);
            await writeFile(path.join(toolsDir, 'libGL.so.1'), '');

            const { code, lines } = launch(['chain.chn'], { ...env, LD_LIBRARY_PATH: toolsDir });

            expect(code).toBe(0);
            expect(lines).toEqual(['chain.chn']);
        });

        test('CHAINNER_SKIP_LIBRARY_CHECK=1 starts the program whatever is missing', async () => {
            await provideLibraries(['libGL.so.1']);

            const { lines, stderr } = launch(['chain.chn'], {
                ...env,
                CHAINNER_SKIP_LIBRARY_CHECK: '1',
            });

            expect(lines).toEqual(['chain.chn']);
            expect(stderr).toBe('');
        });

        describe('running a chain without a display', () => {
            beforeEach(() => provideLibraries());

            test('needs xvfb-run and xauth', () => {
                const { code, lines, stderr } = launch(['run', 'chain.chn'], env);

                expect(code).toBe(1);
                expect(lines).toEqual([]);
                expect(stderr).toContain('this system lacks xvfb-run xauth.\n');
            });

            test('starts under xvfb-run when it and xauth are installed', async () => {
                await writeScript(path.join(toolsDir, 'xvfb-run'), 'printf "%s\\n" xvfb "$@"');
                await writeScript(path.join(toolsDir, 'xauth'), 'exit 0');

                const { lines } = launch(['run', 'chain.chn'], env);

                expect(lines).toEqual([
                    'xvfb',
                    '-a',
                    path.join(appDir, 'chainner'),
                    'run',
                    'chain.chn',
                ]);
            });

            test.each(['DISPLAY', 'WAYLAND_DISPLAY'])('needs neither with %s', (name) => {
                const { lines, stderr } = launch(['run', 'chain.chn'], { ...env, [name]: 'x' });

                expect(lines).toEqual(['run', 'chain.chn']);
                expect(stderr).toBe('');
            });
        });
    }
);
