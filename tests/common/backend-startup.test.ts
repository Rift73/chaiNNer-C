import { ChildProcess, spawn } from 'child_process';
import { existsSync } from 'fs';
import http from 'http';
import net from 'net';
import path from 'path';
import { afterEach, expect, test } from 'vitest';

// The Python that `npm run test:py` uses, and the native module the server needs, as for
// backend-nodes.test.ts.
const python = path.resolve('native/runtime/cpython-3.14.8/python.exe');
const nativeGraph = path.resolve('backend/src/nodes/impl/_chainner_graph.pyd');

const freePort = () =>
    new Promise<number>((resolve) => {
        const server = net.createServer().listen(0, '127.0.0.1', () => {
            const { port } = server.address() as net.AddressInfo;
            server.close(() => resolve(port));
        });
    });

const delay = (ms: number) =>
    new Promise<void>((resolve) => {
        setTimeout(resolve, ms);
    });

let backend: ChildProcess | undefined;
afterEach(() => {
    backend?.kill();
});

// The editor's window reaches the backend through Chromium, which opens at most 6 connections to
// one host, and a request waits for a free one. At a start from the node list cache, the backend
// takes requests about a second after the launch, and the frameworks load for several seconds
// more. A request that needs a node that still loads waits for it in the backend, holding a
// connection, so the window sends a preview only once the backend has registered its node
// (backend-nodes-registered), and fetches the node list, the packages and the features, which
// wait for every node, only once all have loaded (backend-nodes-loaded). Meanwhile it holds just
// the setup events (/setup-sse) and the execution events (/sse). Before, the 3 lists and Load
// Model's preview held the rest, and the startup chain's Load Image, a node that had loaded,
// waited for a free connection until every node had loaded: about 12 s after launch, not 2 s.
test.skipIf(!existsSync(python) || !existsSync(nativeGraph))(
    "A light node's preview at startup does not wait for the frameworks to load",
    async () => {
        const port = await freePort();
        backend = spawn(python, ['-B', 'run.py', String(port)], {
            cwd: path.resolve('backend/src'),
            // starting the backend starts no GPU
            env: {
                ...process.env,
                CUDA_VISIBLE_DEVICES: '-1',
                VK_LOADER_DRIVERS_DISABLE: '*',
                PYTHONNOUSERSITE: '1',
            },
            stdio: 'ignore',
        });
        const agent = new http.Agent({ keepAlive: true, maxSockets: 6 });

        const post = (route: string, body: unknown) =>
            new Promise<unknown>((resolve, reject) => {
                const payload = JSON.stringify(body);
                const req = http.request(
                    {
                        host: '127.0.0.1',
                        port,
                        method: 'POST',
                        path: route,
                        agent,
                        headers: {
                            'Content-Type': 'application/json',
                            'Content-Length': Buffer.byteLength(payload),
                        },
                    },
                    (res) => {
                        let data = '';
                        res.on('data', (chunk) => {
                            data += String(chunk);
                        });
                        res.on('end', () => resolve(JSON.parse(data)));
                        res.on('error', reject);
                    }
                );
                req.on('error', reject);
                req.end(payload);
            });
        const events: string[] = [];
        // the schema ids of each backend-nodes-registered, in order
        const registered: string[][] = [];
        const stream = (route: string) =>
            new Promise<void>((resolve, reject) => {
                http.get({ host: '127.0.0.1', port, path: route, agent }, (res) => {
                    resolve();
                    let buffer = '';
                    res.on('data', (chunk) => {
                        buffer += String(chunk);
                        const blocks = buffer.split('\n\n');
                        buffer = blocks.pop() ?? '';
                        for (const block of blocks) {
                            const event = /^event: (.*)$/m.exec(block)?.[1] ?? '';
                            events.push(event);
                            if (event === 'backend-nodes-registered') {
                                const data = /^data: (.*)$/m.exec(block)?.[1] ?? '{}';
                                registered.push(
                                    (JSON.parse(data) as { schemaIds: string[] }).schemaIds
                                );
                            }
                        }
                    });
                    res.on('error', () => {});
                }).on('error', reject);
            });
        // Bounded, so a backend that does not start fails the test instead of holding the gate.
        const deadline = Date.now() + 60_000;
        const waitUntil = (condition: () => boolean, what: string) =>
            new Promise<void>((resolve, reject) => {
                const check = () => {
                    if (condition()) {
                        resolve();
                    } else if (Date.now() > deadline) {
                        reject(new Error(`The backend did not send ${what} within 60 s`));
                    } else {
                        setTimeout(check, 10);
                    }
                };
                check();
            });

        // as the window does: tried again every 100 ms until the backend listens
        const connect = (route: string): Promise<void> =>
            stream(route).catch(async (error: unknown) => {
                if (Date.now() > deadline) {
                    throw error;
                }
                await delay(100);
                return connect(route);
            });
        await connect('/setup-sse');
        await waitUntil(() => events.includes('backend-ready'), 'backend-ready');
        stream('/sse').catch(() => {});

        // The window holds the previews of the startup chain's PyTorch nodes until PyTorch's
        // nodes are announced, and sends the math node's once the light nodes are.
        await waitUntil(() => registered.length > 0, 'backend-nodes-registered');
        expect(registered[0]).toContain('chainner:utility:math');
        expect(registered[0]).not.toContain('chainner:pytorch:load_model');
        const sent = Date.now();
        await expect(
            post('/run/individual', {
                id: 'math',
                schemaId: 'chainner:utility:math',
                inputs: ['add', 1, 2],
                options: {},
            })
        ).resolves.toEqual({ success: true, data: null });
        expect(Date.now() - sent).toBeLessThan(3000);
    },
    120_000
);
