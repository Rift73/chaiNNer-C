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
        const get = (route: string) =>
            new Promise<number>((resolve, reject) => {
                const start = Date.now();
                http.get({ host: '127.0.0.1', port, path: route, agent }, (res) => {
                    res.resume();
                    res.on('end', () => resolve(Date.now() - start));
                    res.on('error', reject);
                }).on('error', reject);
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
        const waitUntil = async (condition: () => boolean) => {
            while (!condition()) {
                // eslint-disable-next-line no-await-in-loop
                await delay(10);
            }
        };
        const isRegistered = (schemaId: string) =>
            events.includes('backend-nodes-loaded') ||
            registered.some((schemaIds) => schemaIds.includes(schemaId));
        // as the window does: a preview once the backend has its node
        const preview = async (id: string, schemaId: string, inputs: unknown[]) => {
            await waitUntil(() => isRegistered(schemaId));
            const sent = Date.now();
            const result = await post('/run/individual', { id, schemaId, inputs, options: {} });
            return { result, ms: Date.now() - sent };
        };

        // as the window does: tried again every 100 ms until the backend listens
        for (;;) {
            try {
                // eslint-disable-next-line no-await-in-loop
                await stream('/setup-sse');
                break;
            } catch {
                // eslint-disable-next-line no-await-in-loop
                await delay(100);
            }
        }
        await waitUntil(() => events.includes('backend-ready'));
        stream('/sse').catch(() => {});

        // a startup chain with more models than the window has connections, and a math node
        const models = [1, 2, 3, 4, 5, 6].map((i) =>
            preview(`model${i}`, 'chainner:pytorch:load_model', ['C:/nonexistent/model.pth'])
        );
        const math = await preview('math', 'chainner:utility:math', ['add', 1, 2]);
        expect(math.result).toEqual({ success: true, data: null });
        expect(math.ms).toBeLessThan(3000);
        // the light nodes come first, PyTorch's later
        expect(registered[0]).toContain('chainner:utility:math');
        expect(registered[0]).not.toContain('chainner:pytorch:load_model');

        for (const model of await Promise.all(models)) {
            // the file does not exist; the node had loaded when its preview was sent
            expect(model.result).toMatchObject({ success: false });
            expect(model.ms).toBeLessThan(3000);
        }

        // the window then fetches the lists, which no longer wait
        await waitUntil(() => events.includes('backend-nodes-loaded'));
        expect(registered.at(-1)).toContain('chainner:pytorch:load_model');
        for (const route of ['/nodes', '/packages', '/features']) {
            // eslint-disable-next-line no-await-in-loop
            expect(await get(route)).toBeLessThan(2000);
        }
    },
    300_000
);
