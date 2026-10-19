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
// more. What the window holds meanwhile: the setup events (/setup-sse), the execution events
// (/sse) and each preview of a node whose framework still loads, which waits in the backend. The
// node list, the packages and the features also wait for every node, so the window fetches them
// once the backend says its nodes have loaded (backend-nodes-loaded). Fetched earlier, they held
// 3 more connections, and the preview of the startup chain's Load Image, a node that had loaded,
// waited for a free one until every node had loaded: about 12 s after the launch, not 2 s.
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
        const stream = (route: string, onEvent: (event: string) => void) =>
            new Promise<void>((resolve, reject) => {
                http.get({ host: '127.0.0.1', port, path: route, agent }, (res) => {
                    resolve();
                    res.on('data', (chunk) => {
                        for (const match of String(chunk).matchAll(/event: (\S+)/g)) {
                            onEvent(match[1]);
                        }
                    });
                    res.on('error', () => {});
                }).on('error', reject);
            });
        const waitFor = async (event: string) => {
            while (!events.includes(event)) {
                // eslint-disable-next-line no-await-in-loop
                await delay(10);
            }
        };

        // as the window does: tried again every 100 ms until the backend listens
        for (;;) {
            try {
                // eslint-disable-next-line no-await-in-loop
                await stream('/setup-sse', (event) => events.push(event));
                break;
            } catch {
                // eslint-disable-next-line no-await-in-loop
                await delay(100);
            }
        }
        await waitFor('backend-ready');
        stream('/sse', () => {}).catch(() => {});

        // the startup chain's previews: Load Model waits for PyTorch, the math node has loaded
        const loadModel = post('/run/individual', {
            id: 'model',
            schemaId: 'chainner:pytorch:load_model',
            inputs: ['C:/nonexistent/model.pth'],
            options: {},
        });
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
        expect(events).not.toContain('backend-nodes-loaded');

        // the window then fetches the lists, which no longer wait
        await waitFor('backend-nodes-loaded');
        for (const route of ['/nodes', '/packages', '/features']) {
            // eslint-disable-next-line no-await-in-loop
            expect(await get(route)).toBeLessThan(2000);
        }
        await expect(loadModel).resolves.toMatchObject({ success: false });
    },
    300_000
);
