import { execFileSync } from 'node:child_process';
import { builtinModules } from 'node:module';
import os from 'node:os';
import type { AddressInfo } from 'node:net';
import type { ConfigEnv, Plugin, UserConfig } from 'vite';
import './forge-types';

export const builtins = ['electron', ...builtinModules.map((m) => [m, `node:${m}`]).flat()];

export const external = [...builtins, ...(os.platform() === 'darwin' ? ['fsevents'] : [])];

export const getBuildConfig = (env: ConfigEnv<'build'>): UserConfig => {
    const { root, mode, command } = env;

    return {
        root,
        mode,
        build: {
            // Prevent multiple builds from interfering with each other.
            emptyOutDir: false,
            // 🚧 Multiple builds may conflict.
            outDir: '.vite/build',
            watch: command === 'serve' ? {} : null,
            minify: command === 'build',
        },
        clearScreen: false,
    };
};

export const getDefineKeys = (names: string[]) => {
    const define: { [name: string]: VitePluginRuntimeKeys } = {};

    return names.reduce((acc, name) => {
        const NAME = name.toUpperCase();
        const keys: VitePluginRuntimeKeys = {
            VITE_DEV_SERVER_URL: `${NAME}_VITE_DEV_SERVER_URL`,
            VITE_NAME: `${NAME}_VITE_NAME`,
        };

        return { ...acc, [name]: keys };
    }, define);
};

export const getBuildDefine = (env: ConfigEnv<'build'>) => {
    const { command, forgeConfig } = env;
    const names = forgeConfig.renderer.filter(({ name }) => name != null).map(({ name }) => name!);
    const defineKeys = getDefineKeys(names);
    const define = Object.entries(defineKeys).reduce((acc, [name, keys]) => {
        const { VITE_DEV_SERVER_URL, VITE_NAME } = keys;
        const def = {
            [VITE_DEV_SERVER_URL]:
                command === 'serve' ? JSON.stringify(process.env[VITE_DEV_SERVER_URL]) : undefined,
            [VITE_NAME]: JSON.stringify(name),
        };
        return { ...acc, ...def };
    }, {} as Record<string, any>);

    return define;
};

// The fingerprint of the commit the build is made from (BuildInfo in src/common/version.ts): its
// tree, its commit date and whether tracked files have uncommitted changes (untracked files, such
// as local build folders, are not part of the build's commit). The build prints it once, so a
// release build made from a dirty checkout shows in the make log.
export const getBuildInfoDefine = (root: string) => {
    const git = (...args: string[]) =>
        execFileSync('git', args, {
            cwd: root,
            encoding: 'utf8',
            stdio: ['ignore', 'pipe', 'ignore'],
        }).trim();
    const define = (tree: string, date: string, dirty: boolean) => {
        process.stdout.write(
            `Build fingerprint: BUILD_TREE=${tree} BUILD_DATE=${date} BUILD_DIRTY=${String(
                dirty
            )}\n`
        );
        return {
            BUILD_TREE: JSON.stringify(tree),
            BUILD_DATE: JSON.stringify(date),
            BUILD_DIRTY: JSON.stringify(dirty),
        };
    };

    try {
        return define(
            git('rev-parse', 'HEAD^{tree}'),
            git('log', '-1', '--format=%cI'),
            git('status', '--porcelain', '--untracked-files=no') !== ''
        );
    } catch {
        // not a git checkout, or no git: empty values, a development build
        return define('', '', false);
    }
};

export const pluginExposeRenderer = (name: string): Plugin => {
    const { VITE_DEV_SERVER_URL } = getDefineKeys([name])[name];

    return {
        name: '@electron-forge/plugin-vite:expose-renderer',
        configureServer: (server) => {
            process.viteDevServers ??= {};
            // Expose server for preload scripts hot reload.
            process.viteDevServers[name] = server;

            server.httpServer?.once('listening', () => {
                const addressInfo = server.httpServer!.address() as AddressInfo;
                // Expose env constant for main process use.
                process.env[VITE_DEV_SERVER_URL] = `http://localhost:${addressInfo?.port}`;
            });
        },
    };
};

export const pluginHotRestart = (command: 'reload' | 'restart'): Plugin => ({
    name: '@electron-forge/plugin-vite:hot-restart',
    closeBundle: () => {
        if (command === 'reload') {
            // eslint-disable-next-line no-restricted-syntax
            for (const server of Object.values(process.viteDevServers)) {
                // Preload scripts hot reload.
                server.ws.send({ type: 'full-reload' });
            }
        } else {
            // Main process hot restart.
            // https://github.com/electron/forge/blob/v7.2.0/packages/api/core/src/api/start.ts#L216-L223
            process.stdin.emit('data', 'rs');
        }
    },
});
