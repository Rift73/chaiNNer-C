import { afterEach, expect, test, vi } from 'vitest';

afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
});

test('sanitizedEnv drops the variables that move Python to another installation', async () => {
    const otherPython = 'C:\\Python312\\DLLs';
    vi.stubEnv('PYTHONHOME', otherPython);
    vi.stubEnv('PYTHONPATH', otherPython);
    vi.stubEnv('PYTHONPLATLIBDIR', 'lib64');
    vi.stubEnv('PYTHONSAFEPATH', '1');
    vi.stubEnv('PYTHONNOUSERSITE', '');

    const { sanitizedEnv } = await import('../../src/main/env');

    expect(sanitizedEnv.PYTHONHOME).toBeUndefined();
    expect(sanitizedEnv.PYTHONPATH).toBeUndefined();
    expect(sanitizedEnv.PYTHONPLATLIBDIR).toBeUndefined();
    expect(sanitizedEnv.PYTHONSAFEPATH).toBeUndefined();
    expect(sanitizedEnv.PYTHONNOUSERSITE).toBe('1');
});
