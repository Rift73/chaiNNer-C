import { afterEach, expect, test, vi } from 'vitest';
import { BuildInfo } from '../../src/common/version';
import {
    GitHubRelease,
    getLatestVersionIfUpdateAvailable,
    getReleaseTitle,
    getUpdateKey,
    isUpdateAvailable,
} from '../../src/renderer/helpers/github';

const treeR7 = '7'.repeat(40);
const treeR8 = '8'.repeat(40);
const treeR9 = '9'.repeat(40);

// re-release 7 of v0.3.5, as `git log -1 --format=%cI` gives its date: 2026-10-16 04:00 UTC
const build: BuildInfo = {
    tree: treeR7,
    date: '2026-10-16T12:00:00+08:00',
    dirty: false,
    packaged: true,
};

const release = (version: GitHubRelease['version'], tree: string, date: string): GitHubRelease => ({
    version,
    name: `chaiNNer-C v${version}`,
    url: `https://github.com/Rift73/chaiNNer-C/releases/tag/v${version}`,
    tree,
    date,
});

const r8 = release('0.3.5', treeR8, '2026-10-18T03:00:00Z');

// for chaiNNer-C v0.3.5
const isUpdate = (latest: GitHubRelease, current: BuildInfo = build) =>
    isUpdateAvailable(latest, '0.3.5', current);

test('a newer version is an update', () => {
    expect(isUpdate(release('0.3.6', treeR8, '2026-10-20T00:00:00Z'))).toBe(true);
    expect(isUpdate(release('0.4.0', treeR7, '2026-10-01T00:00:00Z'))).toBe(true);
});

test('the same version with another tree, committed later, is an update', () => {
    expect(isUpdate(r8)).toBe(true);
    // dates are compared as instants: 05:00 UTC is an hour after 12:00 at +08:00
    expect(isUpdate(release('0.3.5', treeR8, '2026-10-16T05:00:00Z'))).toBe(true);
});

test('the same tree is not an update', () => {
    expect(isUpdate(release('0.3.5', treeR7, '2026-10-18T03:00:00Z'))).toBe(false);
});

test('another tree committed earlier is not an update', () => {
    expect(isUpdate(release('0.3.5', treeR8, '2026-10-16T03:00:00Z'))).toBe(false);
});

test('an older version is not an update', () => {
    expect(isUpdate(release('0.3.4', treeR8, '2026-10-18T03:00:00Z'))).toBe(false);
});

test('a development build only counts newer versions', () => {
    const newer = release('0.3.6', treeR8, '2026-10-20T00:00:00Z');
    const developmentBuilds: BuildInfo[] = [
        { ...build, dirty: true },
        { ...build, packaged: false },
        // built outside git
        { ...build, tree: '', date: '' },
    ];
    for (const developmentBuild of developmentBuilds) {
        expect(isUpdate(r8, developmentBuild)).toBe(false);
        expect(isUpdate(newer, developmentBuild)).toBe(true);
    }
});

test('a release is ignored by its version and tree', () => {
    expect(getUpdateKey(r8)).toBe(`0.3.5@${treeR8}`);
    // ignoring re-release 8 does not ignore re-release 9
    const r9 = release('0.3.5', treeR9, '2026-10-25T03:00:00Z');
    expect(getUpdateKey(r9)).not.toBe(getUpdateKey(r8));
});

test('the title of a re-release has its date', () => {
    expect(getReleaseTitle(r8, '0.3.5')).toBe('chaiNNer-C v0.3.5, re-released 2026-10-18');
    expect(getReleaseTitle(release('0.3.6', treeR8, '2026-10-20T00:00:00Z'), '0.3.5')).toBe(
        'chaiNNer-C v0.3.6'
    );
});

const api = 'https://api.github.com/repos/Rift73/chaiNNer-C';
const latestRelease = {
    tag_name: 'v0.3.5',
    name: 'chaiNNer-C v0.3.5',
    html_url: 'https://github.com/Rift73/chaiNNer-C/releases/tag/v0.3.5',
};
const json = (data: unknown) => new Response(JSON.stringify(data));
const tagCommit = (tree: string) => ({
    commit: { tree: { sha: tree }, committer: { date: '2026-10-18T03:00:00Z' } },
});

const mockFetch = (responses: Partial<Record<string, () => Response>>) => {
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
        expect(init?.signal).toBeInstanceOf(AbortSignal);
        const response = responses[url];
        return Promise.resolve(response ? response() : new Response('', { status: 404 }));
    });
    vi.stubGlobal('fetch', fetchMock);
    return fetchMock;
};

afterEach(() => {
    vi.unstubAllGlobals();
});

test('the check reads the latest release and the commit of its tag', async () => {
    const fetchMock = mockFetch({
        [`${api}/releases/latest`]: () => json(latestRelease),
        [`${api}/commits/v0.3.5`]: () => json(tagCommit(treeR8)),
    });

    await expect(getLatestVersionIfUpdateAvailable('0.3.5', build)).resolves.toEqual({
        version: '0.3.5',
        name: 'chaiNNer-C v0.3.5',
        url: latestRelease.html_url,
        tree: treeR8,
        date: '2026-10-18T03:00:00Z',
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    // both requests share one 5 s timeout
    const [[, latestInit], [, commitInit]] = fetchMock.mock.calls;
    expect(commitInit?.signal).toBe(latestInit?.signal);
});

test('the check finds no update for the same tree', async () => {
    mockFetch({
        [`${api}/releases/latest`]: () => json(latestRelease),
        [`${api}/commits/v0.3.5`]: () => json(tagCommit(treeR7)),
    });

    await expect(getLatestVersionIfUpdateAvailable('0.3.5', build)).resolves.toBeNull();
});

test('the check fails on a rate limit, a non-2xx status or a response it cannot parse', async () => {
    mockFetch({
        [`${api}/releases/latest`]: () =>
            new Response('{"message":"API rate limit exceeded"}', {
                status: 403,
                statusText: 'rate limit exceeded',
            }),
    });
    await expect(getLatestVersionIfUpdateAvailable('0.3.5', build)).rejects.toThrow('403');

    mockFetch({
        [`${api}/releases/latest`]: () => json(latestRelease),
    });
    await expect(getLatestVersionIfUpdateAvailable('0.3.5', build)).rejects.toThrow('404');

    mockFetch({
        [`${api}/releases/latest`]: () => new Response('<html>Unicorn!</html>'),
    });
    await expect(getLatestVersionIfUpdateAvailable('0.3.5', build)).rejects.toThrow();

    mockFetch({
        [`${api}/releases/latest`]: () => json({ ...latestRelease, tag_name: 'latest' }),
        [`${api}/commits/latest`]: () => json(tagCommit(treeR8)),
    });
    await expect(getLatestVersionIfUpdateAvailable('0.3.5', build)).rejects.toThrow(
        "Invalid version 'latest'"
    );

    mockFetch({
        [`${api}/releases/latest`]: () => json(latestRelease),
        [`${api}/commits/v0.3.5`]: () =>
            json({ commit: { tree: { sha: treeR8 }, committer: null } }),
    });
    await expect(getLatestVersionIfUpdateAvailable('0.3.5', build)).rejects.toThrow('no date');
});

test('the check fails when it is offline', async () => {
    vi.stubGlobal(
        'fetch',
        vi.fn(() => Promise.reject(new TypeError('Failed to fetch')))
    );

    await expect(getLatestVersionIfUpdateAvailable('0.3.5', build)).rejects.toThrow(
        'Failed to fetch'
    );
});
