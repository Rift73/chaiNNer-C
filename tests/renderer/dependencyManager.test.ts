import { expect, test } from 'vitest';
import { Package, PackageId, PyPiName, PyPiPackage, Version } from '../../src/common/common-types';
import {
    PackageUpdateInfo,
    formatBytes,
    formatSizeEstimate,
    getAllOperation,
} from '../../src/renderer/helpers/dependencyManager';

const MB = 1024 ** 2;

const dep = (pypiName: string, sizeEstimate: number, url?: string): PyPiPackage => ({
    displayName: pypiName,
    pypiName: pypiName as PyPiName,
    version: '1.0.0' as Version,
    url,
    sizeEstimate,
    autoUpdate: false,
});

const rocmWindows = (): PyPiPackage[] => [
    dep('rocm', 0.01 * MB, 'file:///rocm-7.14.1+chainner-py3-none-any.whl'),
    dep('rocm-sdk-core', 400 * MB, 'https://example.test/rocm_sdk_core.whl#sha256=aa'),
    dep('rocm-sdk-libraries', 100 * MB, 'https://example.test/rocm_sdk_libraries.whl#sha256=bb'),
    dep('rocm-sdk-device-gfx1100', 50 * MB, 'https://example.test/rocm_sdk_device.whl#sha256=cc'),
];

const pkg = (id: string, dependencies: PyPiPackage[], disabled?: boolean): Package => ({
    id: id as PackageId,
    name: id,
    icon: '',
    color: '',
    description: '',
    dependencies,
    features: [],
    settings: [],
    disabled,
});

/** What the All button of the packages says it downloads, with `installed` already installed. */
const allOperation = (packages: Package[], installed: PyPiPackage[], version = '1.0.0') => {
    const installedPyPi: Partial<Record<PyPiName, Version>> = {};
    for (const d of installed) installedPyPi[d.pypiName] = version as Version;
    const operation = getAllOperation(
        packages,
        packages.map((p) => new PackageUpdateInfo(p, installedPyPi))
    );
    return {
        ids: operation.packages.map((p) => p.id),
        size: formatSizeEstimate(operation.dependencies),
    };
};

test('formatBytes picks the unit and rounds to two digits', () => {
    expect(formatBytes(500)).toBe('500\u2009Byte');
    expect(formatBytes(1536)).toBe('1.5\u2009KB');
    expect(formatBytes(580 * 1024)).toBe('580\u2009KB');
    expect(formatBytes(45 * MB)).toBe('45\u2009MB');
    expect(formatBytes(1536 * MB)).toBe('1.5\u2009GB');
    expect(formatBytes(2 * 1024 * MB)).toBe('2\u2009GB');
});

test('a dependency several packages list counts once in the total', () => {
    // The PyTorch, ONNX and MIGraphX packages of an AMD GPU list the same ROCm wheels.
    const pytorch = [dep('torch', 1000 * MB), dep('torchvision', 10 * MB), ...rocmWindows()];
    const onnx = [dep('onnx', 20 * MB), dep('onnxruntime-migraphx', 70 * MB), ...rocmWindows()];
    const migraphx = [dep('migraphx-tools', 30 * MB), ...rocmWindows()];
    const everything = [pytorch, onnx, migraphx].flat();

    // 1000 + 10 + 20 + 70 + 30 + the ROCm wheels (550 MB) once = 1680 MB, not 2780 MB
    expect(formatSizeEstimate(everything)).toBe('1.6\u2009GB');
    expect(formatSizeEstimate([...pytorch, ...onnx])).toBe(formatBytes(1650 * MB));
    expect(formatSizeEstimate([...pytorch, ...onnx, ...migraphx])).toBe(formatBytes(1680 * MB));
});

test('a dependency counts once whatever the spelling of its name', () => {
    const extras = [dep('vsjetpack[deinterlace]', 40 * MB), dep('vsjetpack', 40 * MB)];
    expect(formatSizeEstimate(extras)).toBe(formatBytes(40 * MB));

    const spellings = [
        dep('onnx_graphsurgeon', 10 * MB),
        dep('Onnx-GraphSurgeon', 10 * MB),
        dep('onnx.graphsurgeon', 10 * MB),
    ];
    expect(formatSizeEstimate(spellings)).toBe(formatBytes(10 * MB));
});

test('packages that share nothing sum every dependency', () => {
    const pytorch = [dep('torch', 2048 * MB), dep('torchvision', 2 * MB), dep('einops', 42)];
    const onnx = [dep('onnxruntime-gpu', 200 * MB), dep('onnx', 15 * MB), dep('protobuf', 1 * MB)];
    const tensorrt = [
        dep('tensorrt', 1200 * MB),
        dep('cuda-python', 20 * MB),
        dep('triton', 250 * MB),
    ];

    expect(formatSizeEstimate([...pytorch, ...onnx, ...tensorrt])).toBe(
        formatBytes(2048 * MB + 2 * MB + 42 + 200 * MB + 15 * MB + 1 * MB + 1470 * MB)
    );
    // one package's own Install or Update button
    expect(formatSizeEstimate(pytorch)).toBe(formatBytes(2050 * MB + 42));
    expect(formatSizeEstimate([])).toBe('0\u2009Byte');
});

test('the All button counts what it installs, not what is already installed', () => {
    // PyTorch was installed first: its ROCm wheels are there when ONNX comes.
    const pytorchDeps = [dep('torch', 1000 * MB), dep('torchvision', 10 * MB), ...rocmWindows()];
    const onnxDeps = [dep('onnx', 20 * MB), dep('onnxruntime-migraphx', 70 * MB), ...rocmWindows()];
    const pytorch = pkg('chaiNNer_pytorch', pytorchDeps);
    const onnx = pkg('chaiNNer_onnx', onnxDeps);

    // the ONNX package's own Install button: 20 + 70 MB
    const info = new PackageUpdateInfo(
        onnx,
        Object.fromEntries(pytorchDeps.map((d) => [d.pypiName, d.version]))
    );
    expect(formatSizeEstimate([...info.missing, ...info.outdated])).toBe(formatBytes(90 * MB));

    expect(allOperation([pytorch, onnx], pytorchDeps)).toEqual({
        ids: ['chaiNNer_onnx'],
        size: formatBytes(90 * MB),
    });

    // nothing installed: both packages, the ROCm wheels once (1000 + 10 + 20 + 70 + 550 MB)
    expect(allOperation([pytorch, onnx], [])).toEqual({
        ids: ['chaiNNer_pytorch', 'chaiNNer_onnx'],
        size: formatBytes(1650 * MB),
    });
});

test('the All button counts the missing and outdated dependencies of a package', () => {
    const packages = [
        pkg('a', [dep('current', 100 * MB), dep('old', 30 * MB), dep('absent', 5 * MB)]),
        pkg('b', [dep('current-too', 200 * MB)]),
    ];
    const installed = {
        current: '1.0.0',
        old: '0.9.0',
        'current-too': '1.2.0',
    } as Partial<Record<PyPiName, Version>>;
    const operation = getAllOperation(
        packages,
        packages.map((p) => new PackageUpdateInfo(p, installed))
    );

    // package b has nothing to do; a updates `old` and installs `absent`
    expect(operation.packages.map((p) => p.id)).toEqual(['a']);
    expect(formatSizeEstimate(operation.dependencies)).toBe(formatBytes(35 * MB));
});

test('the All button leaves out disabled packages', () => {
    const packages = [
        pkg('enabled', [dep('x', 10 * MB)]),
        pkg('disabled', [dep('y', 500 * MB)], true),
    ];
    expect(allOperation(packages, [])).toEqual({ ids: ['enabled'], size: formatBytes(10 * MB) });
});
