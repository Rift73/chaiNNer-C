import { execFileSync } from 'child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'fs';
import os from 'os';
import path from 'path';
import { expect, test } from 'vitest';
import { BackendNodesResponse } from '../../src/common/Backend';
import { CategoryMap } from '../../src/common/CategoryMap';
import { parseFunctionDefinitions } from '../../src/common/nodes/parseFunctionDefinitions';
import { sortNodes } from '../../src/common/nodes/sort';
import { PassthroughMap } from '../../src/common/PassthroughMap';
import { SchemaInputsMap } from '../../src/common/SchemaInputsMap';
import { SchemaMap } from '../../src/common/SchemaMap';

// The Python that `npm run test:py` uses, and the native module the server needs. Both exist
// only where the runtime is provisioned and the native code is built, as for backend/tests.
const python = path.resolve('native/runtime/cpython-3.14.8/python.exe');
const nativeGraph = path.resolve('backend/src/nodes/impl/_chainner_graph.pyd');

// Writes the server's /nodes reply to argv[1]. The packages load as on every start: a node
// module whose optional dependency is missing is left out, and the rest still register.
const DUMP_NODES = `
import asyncio, os, sys
out = sys.argv.pop()  # server.py parses the command line when it is imported
sys.path.insert(0, os.getcwd())
import server
from server_config import ServerConfig
config = ServerConfig(
    port=0,
    close_after_start=False,
    install_builtin_packages=False,
    error_on_failed_node=False,
    storage_dir=None,
    trace=False,
)
asyncio.run(server.import_packages(config))
with open(out, "wb") as f:
    f.write(asyncio.run(server.nodes(None)).body)
`;

const dumpNodes = (): BackendNodesResponse => {
    const dir = mkdtempSync(path.join(os.tmpdir(), 'chaiNNer-nodes-'));
    try {
        const file = path.join(dir, 'nodes.json');
        execFileSync(python, ['-B', '-c', DUMP_NODES, file], {
            cwd: path.resolve('backend/src'),
            // listing the nodes starts no GPU
            env: { ...process.env, CUDA_VISIBLE_DEVICES: '-1', VK_LOADER_DRIVERS_DISABLE: '*' },
            stdio: ['ignore', 'ignore', 'pipe'],
        });
        return JSON.parse(readFileSync(file, 'utf-8')) as BackendNodesResponse;
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
};

// One node type the frontend cannot evaluate makes it reject the whole node list, and every
// start shows "Unable to process backend nodes.". This runs what BackendContext runs on the
// reply (processBackendResponse and the SchemaInputsMap) over the real schemas.
test.skipIf(!existsSync(python) || !existsSync(nativeGraph))(
    'The frontend accepts every backend node',
    () => {
        const { nodes, categories } = dumpNodes();
        expect(nodes.length).toBeGreaterThan(100);

        const categoryMap = new CategoryMap(categories);
        const schemata = new SchemaMap(sortNodes(nodes, categoryMap));
        const functionDefinitions = parseFunctionDefinitions(nodes);
        PassthroughMap.create(functionDefinitions);
        expect(() => new SchemaInputsMap(schemata.schemata)).not.toThrow();

        expect(functionDefinitions.size).toBe(nodes.length);
    },
    300_000
);
