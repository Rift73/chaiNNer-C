import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { expect, test } from 'vitest';
import { currentMigration } from '../../src/common/migrations';
import { RawSaveFile, SaveFile } from '../../src/main/SaveFile';

const dataDir = path.join(__dirname, '..', 'data');

test('Read save file with a UTF-8 byte order mark', async () => {
    // Windows PowerShell 5.1 writes one with `Set-Content -Encoding utf8`
    const filePath = path.join(dataDir, 'crop.chn');
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'save-file-test-'));
    try {
        const bomPath = path.join(dir, 'crop.chn');
        fs.writeFileSync(bomPath, `\uFEFF${fs.readFileSync(filePath, 'utf-8')}`, 'utf-8');

        expect(await SaveFile.read(bomPath)).toEqual(await SaveFile.read(filePath));
    } finally {
        fs.rmSync(dir, { recursive: true, force: true });
    }
});

test('Load Image Pairs index migrates to one edge per input', () => {
    const pairs = '00000000-0000-0000-0000-000000000001';
    const math = '00000000-0000-0000-0000-000000000002';
    const content = {
        nodes: [
            {
                id: pairs,
                type: 'generator',
                position: { x: 0, y: 0 },
                data: {
                    id: pairs,
                    schemaId: 'chainner:image:load_image_pairs',
                    inputData: { 0: 'a', 1: 'b', 2: 0, 3: 10, 4: 0 },
                },
            },
            {
                id: math,
                type: 'regularNode',
                position: { x: 400, y: 0 },
                data: { id: math, schemaId: 'chainner:utility:math', inputData: { 1: 'add' } },
            },
        ],
        edges: [
            {
                id: 'index-edge',
                source: pairs,
                sourceHandle: `${pairs}-8`,
                target: math,
                targetHandle: `${math}-0`,
                type: 'main',
            },
        ],
        viewport: { x: 0, y: 0, zoom: 1 },
    };
    // a v0.24 chain: the last migration splits Load Image Pairs into two Load Images
    const parsed = SaveFile.parse(
        JSON.stringify({ version: '0.24.0', content, migration: currentMigration - 1 })
    );

    const toMath = parsed.edges.filter((e) => e.target === math);
    expect(toMath).toHaveLength(1);
    expect(parsed.nodes.find((n) => n.id === toMath[0].source)?.data.nodeName).toBe(
        'Load Images A'
    );
    expect(toMath[0].sourceHandle).toBe(`${toMath[0].source}-4`);
});

for (const file of fs.readdirSync(dataDir)) {
    const filePath = path.join(dataDir, file);

    test(`Read save file ${file}`, async () => {
        const parsed = await SaveFile.read(filePath);
        expect(parsed).toMatchSnapshot();
    });
    test(`Write save file ${file}`, async () => {
        const json = SaveFile.stringify(await SaveFile.read(filePath), '0.0.0-test');
        const obj = JSON.parse(json) as RawSaveFile;
        delete obj.migration;
        delete obj.timestamp;
        expect(obj).toMatchSnapshot();
    });
}
