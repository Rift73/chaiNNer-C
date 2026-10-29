import { Package, PyPiName, PyPiPackage, Version } from '../../common/common-types';
import { versionGt } from '../../common/version';

export const formatBytes = (bytes: number): string => {
    const KB = 1024 ** 1;
    const MB = 1024 ** 2;
    const GB = 1024 ** 3;

    const rounded = (n: number): string => {
        if (n < 10) return Number(n.toPrecision(2)).toString();
        return n.toFixed(0);
    };

    if (bytes < KB) return `${bytes}\u2009Byte`;
    if (bytes < MB) return `${rounded(bytes / KB)}\u2009KB`;
    if (bytes < GB) return `${rounded(bytes / MB)}\u2009MB`;
    return `${rounded(bytes / GB)}\u2009GB`;
};

export class PackageUpdateInfo {
    readonly missing: readonly PyPiPackage[];

    readonly outdated: readonly PyPiPackage[];

    get canInstall(): boolean {
        return this.missing.length > 0;
    }

    get canUpdate(): boolean {
        return this.outdated.length > 0;
    }

    constructor(pkg: Package, installedPyPi: Readonly<Partial<Record<PyPiName, Version>>>) {
        const missing: PyPiPackage[] = [];
        const outdated: PyPiPackage[] = [];

        for (const dep of pkg.dependencies) {
            const installed = installedPyPi[dep.pypiName];
            if (!installed) {
                missing.push(dep);
            } else if (versionGt(dep.version, installed)) {
                outdated.push(dep);
            }
        }

        this.missing = missing;
        this.outdated = outdated;
    }
}

/**
 * What the All button installs or updates: the packages that are not disabled and have something
 * to install or update, and the dependencies it downloads. Like the packages' own buttons and the
 * backend, it leaves out the dependencies that are installed and up to date.
 */
export const getAllOperation = (
    packages: readonly Package[],
    infos: readonly PackageUpdateInfo[]
): { packages: Package[]; dependencies: PyPiPackage[] } => {
    const operated = packages.flatMap((p, i) => {
        const info = infos[i];
        return !p.disabled && (info.canInstall || info.canUpdate) ? [{ p, info }] : [];
    });
    return {
        packages: operated.map(({ p }) => p),
        dependencies: operated.flatMap(({ info }) => [...info.missing, ...info.outdated]),
    };
};

/**
 * The name pip lists a dependency by: without the `[extras]` it was installed with, in lower case
 * and with every run of `-`, `_` and `.` as one `-`. This is how the backend names it
 * (`distribution_name` in `dependencies/store.py`), and pip installs one distribution per name,
 * whatever the packages that list it and however they pin it (by version or by URL).
 */
const distributionName = (pypiName: string): string =>
    pypiName
        .split('[', 1)[0]
        .replace(/[-_.]+/g, '-')
        .toLowerCase();

/**
 * The download size estimate of the dependencies. A dependency several packages list (the ROCm
 * wheels of the PyTorch and ONNX packages on AMD GPUs) is downloaded once, so it counts once.
 */
export const formatSizeEstimate = (dependencies: readonly PyPiPackage[]): string => {
    const sizes = new Map<string, number>();
    for (const d of dependencies) {
        const name = distributionName(d.pypiName);
        if (!sizes.has(name)) sizes.set(name, d.sizeEstimate);
    }
    return formatBytes([...sizes.values()].reduce((a, b) => a + b, 0));
};
