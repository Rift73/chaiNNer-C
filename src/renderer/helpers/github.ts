import { Version } from '../../common/common-types';
import { BuildInfo, isDevelopmentBuild, parse, versionGt } from '../../common/version';
import type { RestEndpointMethodTypes } from '@octokit/rest';

// DO NOT use @octokit/rest directly. @octokit/rest is unfortunately a node-only package.
// It uses node modules and other node-only dependencies like node-fetch under the hood,
// which mean that we cannot use in the browser. Instead, we use the fetch API directly.

const repository = 'Rift73/chaiNNer-C';

const fetchApi = async <T>(url: string, signal: AbortSignal): Promise<T> => {
    // fetch data from GitHub REST API
    const response = await fetch(`https://api.github.com${url}`, { signal });
    if (!response.ok) {
        throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
    }
    const data: unknown = await response.json();
    return data as T;
};

const getLatestRelease = async (signal: AbortSignal) => {
    const data: RestEndpointMethodTypes['repos']['getLatestRelease']['response']['data'] =
        await fetchApi(`/repos/${repository}/releases/latest`, signal);
    return data;
};

const getCommit = async (ref: string, signal: AbortSignal) => {
    const data: RestEndpointMethodTypes['repos']['getCommit']['response']['data'] = await fetchApi(
        `/repos/${repository}/commits/${encodeURIComponent(ref)}`,
        signal
    );
    return data;
};

// The latest release and the tree and date of the commit its tag points to.
export interface GitHubRelease {
    version: Version;
    name: string;
    url: string;
    tree: string;
    date: string;
}

// Releases are re-released under the same tag, so the same version is an update when its tree
// differs and it was committed later. A development build's tree means nothing: newer versions
// only.
export const isUpdateAvailable = (
    release: GitHubRelease,
    version: Version,
    build: BuildInfo
): boolean => {
    if (versionGt(release.version, version)) {
        return true;
    }
    return (
        !isDevelopmentBuild(build) &&
        release.version === parse(version) &&
        release.tree !== build.tree &&
        Date.parse(release.date) > Date.parse(build.date)
    );
};

// "Don't show again for this release" stores this key: ignoring one re-release keeps the next.
export const getUpdateKey = (release: GitHubRelease): string =>
    `${release.version}@${release.tree}`;

export const getReleaseTitle = (release: GitHubRelease, version: Version): string =>
    release.version === parse(version)
        ? `${release.name}, re-released ${release.date.slice(0, 10)}`
        : release.name;

export const getLatestVersionIfUpdateAvailable = async (
    version: Version,
    build: BuildInfo
): Promise<GitHubRelease | null> => {
    const signal = AbortSignal.timeout(5000);
    const latestRelease = await getLatestRelease(signal);
    const { commit } = await getCommit(latestRelease.tag_name, signal);
    if (!commit.committer?.date) {
        throw new Error(`The commit of ${latestRelease.tag_name} has no date`);
    }
    const release: GitHubRelease = {
        version: parse(latestRelease.tag_name),
        name: latestRelease.name || latestRelease.tag_name,
        url: latestRelease.html_url,
        tree: commit.tree.sha,
        date: commit.committer.date,
    };
    return isUpdateAvailable(release, version, build) ? release : null;
};
