import { createServer } from 'http';
import { AddressInfo } from 'net';
import { afterAll, beforeAll, expect, test } from 'vitest';
import { Backend } from '../../src/common/Backend';
import { PackageId } from '../../src/common/common-types';

// The host's package routes as server_host.py answers them.
const replies: Partial<Record<string, [number, unknown]>> = {
    '/packages/install': [
        500,
        { status: 'error', message: 'An error occurred while installing dependencies.' },
    ],
    '/packages/uninstall': [200, { status: 'ok' }],
};
const server = createServer((request, response) => {
    const [status, body] = replies[request.url ?? ''] ?? [404, {}];
    response.writeHead(status, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify(body));
});
let backend: Backend;

beforeAll(async () => {
    await new Promise<void>((resolve) => {
        server.listen(0, '127.0.0.1', resolve);
    });
    backend = new Backend(`http://127.0.0.1:${(server.address() as AddressInfo).port}`);
});
afterAll(async () => {
    await new Promise((resolve) => {
        server.close(resolve);
    });
});

const packages = ['chaiNNer_pytorch' as PackageId];

test('A failed install rejects with the backend message', async () => {
    await expect(backend.installPackages(packages)).rejects.toThrow(
        'An error occurred while installing dependencies.'
    );
});

test('A successful change resolves', async () => {
    await expect(backend.uninstallPackages(packages)).resolves.toBeUndefined();
});
