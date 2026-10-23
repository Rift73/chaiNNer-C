// eslint-disable-next-line import/no-extraneous-dependencies
import react from '@vitejs/plugin-react-swc';
// eslint-disable-next-line import/no-extraneous-dependencies
import { defineConfig } from 'vite';
import { pluginExposeRenderer } from './base.config';
import type { ConfigEnv, UserConfig } from 'vite';
import './forge-types';

// https://vitejs.dev/config
export default defineConfig((env) => {
    const forgeEnv = env as ConfigEnv<'renderer'>;
    const { root, mode, forgeConfigSelf } = forgeEnv;
    const name = forgeConfigSelf.name ?? '';

    return {
        root,
        mode,
        base: './',
        build: {
            emptyOutDir: false,
            outDir: `.vite/renderer/${name}`,
            rollupOptions: {
                output: {
                    entryFileNames: `[name].js`,
                    chunkFileNames: `[name].js`,
                    assetFileNames: `[name].[ext]`,
                },
            },
        },
        plugins: [pluginExposeRenderer(name), react()],
        resolve: {
            preserveSymlinks: true,
            alias: {
                path: '@chainner/node-path',
                'electron/common': 'electron',
                'electron/renderer': 'electron',
                'electron-log/renderer': 'electron-log',
            },
        },
        clearScreen: false,
        // The renderer's only page. Without it, Vite's dependency scan reads every HTML file under
        // the root (out/, native/), which delays the main process build that `start` waits for.
        optimizeDeps: {
            entries: ['index.html'],
        },
        server: {
            watch: {
                // The renderer's sources are under src/. The other top-level folders hold the Python
                // backend, native builds, the runtime and venv, and packaged apps: watching them made
                // the dev server open a handle per folder (about 237,000 in a full checkout).
                ignored: [
                    '**/translation.json',
                    '**/native/**',
                    '**/backend/**',
                    '**/out/**',
                    '**/.vite/**',
                    '**/.pytest_cache/**',
                    '**/.ruff_cache/**',
                ],
            },
        },
    } as UserConfig;
});
