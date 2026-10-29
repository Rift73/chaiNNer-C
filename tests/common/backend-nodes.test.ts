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
// module whose optional dependency is not installed is left out, and the rest still register.
// A module that fails otherwise (an error, or a module of the backend it cannot find) would
// be left out alike, so it fails the dump, naming the module and its error.
const DUMP_NODES = `
import asyncio, importlib.util, os, pathlib, sys, types
out = sys.argv.pop()  # server.py parses the command line when it is imported
sys.path.insert(0, os.getcwd())
# The worker runs server.py: modules find the backend's files by its path.
sys.modules["__main__"].__file__ = os.path.abspath("server.py")
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
# The node-load failures are recorded in the storage folder: the dump's own folder.
context = types.SimpleNamespace(config=config, storage_dir=pathlib.Path(os.path.dirname(out)))
asyncio.run(server.import_packages(context))

def not_installed(error):
    if not isinstance(error, ModuleNotFoundError) or not error.name:
        return False
    return importlib.util.find_spec(error.name.partition(".")[0]) is None

failed = [
    f"{e.module}: {e.error!r}"
    for errors in server.node_loading.errors.values()
    for e in errors
    if not not_installed(e.error)
]
if failed:
    sys.exit("Node modules failed to load:\\n" + "\\n".join(failed))
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
