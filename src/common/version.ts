import semver from 'semver';
import { Version } from './common-types';

// chaiNNer-C's product version, shown in the header, About, the startup log and system
// information. app.getVersion() stays upstream's: it stamps saved chains, and migrations read it.
export const chainnerCVersion = '0.3.13';

// The build's fingerprint, embedded at build time (vite/base.config.ts): a re-release keeps its
// tag, so its git tree and commit date tell it apart. A build made outside git has empty values.
export interface BuildInfo {
    tree: string;
    date: string;
    dirty: boolean;
    packaged: boolean;
}

// Not packaged, built with uncommitted changes or outside git: no automatic update check.
export const isDevelopmentBuild = ({ tree, date, dirty, packaged }: BuildInfo): boolean =>
    !packaged || dirty || !tree || !date;

export const parse = (v: string): Version => {
    const version = semver.coerce(v);
    if (!version) {
        throw new SyntaxError(`Invalid version '${v}'`);
    }
    return version.version as Version;
};

export const versionGt = (lhs: Version, rhs: Version): boolean => {
    try {
        return semver.gt(parse(lhs), parse(rhs));
    } catch {
        return false;
    }
};

/**
 * The build labels of platforms' wheels: PyTorch's (2.14.1+cu132, +cpu) and AMD's (2.13.0+rocm10.0.0,
 * 3.8.0+git4cff872c.rocm10.0.0, 1.31.1+rocm10.0.hipgraph). An install of another platform's
 * build reports its own, so a pin with one is compared by its numbers only.
 */
const PLATFORM_LABEL = /^(?:cu\d|cpu|rocm|git)/;

/**
 * Whether an installed version needs the pinned one: it is older, or the pin is a build with a
 * label of ours (spandrel 0.4.2+c23, onnxruntime-gpu 1.30.0+trt11.4, rocm 10.0.0+chainner),
 * which only that exact version is, as the backend decides it (dependencies/store.py). A version
 * label alone does not order versions: 0.4.2+c22 and 0.4.2+c23 are the same numbers.
 * One difference stays: the backend reinstalls on any `+` pin mismatch, while this compares
 * numbers only for platform build labels.
 */
export const isOutdated = (pinned: Version, installed: Version): boolean => {
    if (versionGt(pinned, installed)) return true;
    const plus = pinned.indexOf('+');
    if (plus < 0 || PLATFORM_LABEL.test(pinned.slice(plus + 1))) return false;
    return installed !== pinned;
};

export const versionGte = (lhs: Version, rhs: Version): boolean => {
    try {
        return semver.gte(parse(lhs), parse(rhs));
    } catch {
        return false;
    }
};
