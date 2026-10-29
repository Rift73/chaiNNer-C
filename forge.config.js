// eslint-disable-next-line import/no-extraneous-dependencies
const AdmZip = require('adm-zip');
const nodeFs = require('fs');
const fs = require('fs/promises');
const path = require('path');
const packageJson = require('./package.json');

// cross-zip, which @electron-forge/maker-zip uses (4.0.1 is its latest), clears its output with
// fs.rmdir(path, { recursive: true }), an option Node 25 removed. In this build process such a
// call goes to fs.rm with recursive and force, its documented replacement (rm -rf); every other
// rmdir call is unchanged.
const { rmdir, rmdirSync } = nodeFs;
nodeFs.rmdir = (target, options, callback) => {
    if (options && typeof options === 'object' && options.recursive) {
        nodeFs.rm(
            target,
            { recursive: true, force: true, maxRetries: options.maxRetries },
            callback
        );
    } else {
        rmdir(target, options, callback);
    }
};
nodeFs.rmdirSync = (target, options) => {
    if (options && options.recursive) {
        nodeFs.rmSync(target, { recursive: true, force: true, maxRetries: options.maxRetries });
    } else {
        rmdirSync(target, options);
    }
};

const makerOptions = {
    categories: ['Graphics'],
    genericName: 'Image Processing GUI',
    homepage: 'https://chainner.app',
    icon: './src/public/icons/cross_platform/icon.png',
    mimeType: ['application/json'],
    productDescription:
        'A node-based image processing GUI aimed at making chaining image processing tasks easy and customizable. Born as an AI upscaling application, chaiNNer has grown into an extremely flexible and powerful programmatic image processing application.\n\nChaiNNer gives you a level of customization of your image processing workflow that very few others do. Not only do you have full control over your processing pipeline, you can do incredibly complex tasks just by connecting a few nodes together.',
};

// The native layer's Linux binaries (relative to backend/src) that every package needs, and VOSR's
// TensorRT plugin, which only a package meant to lack the VOSR TensorRT backends may do without.
const linuxNativeBinaries = [
    'nodes/impl/chainner_native.so',
    'nodes/impl/_chainner_graph.so',
    'chainner_ext/chainner_ext.so',
];
const linuxVosrPlugin = 'nodes/impl/pytorch/vosr2_qkv.so';

const deletePycFiles = async (directory) => {
    try {
        const files = await fs.readdir(directory, { withFileTypes: true });
        await Promise.all(
            files.map(async (file) => {
                const fullPath = path.join(directory, file.name);

                if (file.isDirectory()) {
                    if (file.name === '__pycache__') {
                        await fs.rm(fullPath, { recursive: true, force: true });
                    } else {
                        await deletePycFiles(fullPath);
                    }
                } else if (file.isFile() && path.extname(file.name) === '.pyc') {
                    await fs.unlink(fullPath);
                }
            })
        );
    } catch (error) {
        // eslint-disable-next-line no-console
        console.error(error);
    }
};

/** @type {import("@electron-forge/shared-types").ForgeConfig} */
const config = {
    packagerConfig: {
        asar: false,
        executableName: process.platform === 'linux' ? packageJson.name : packageJson.productName,
        extraResource: ['./backend/src/', './src/public/icons/mac/file_chn.icns'],
        icon: makerOptions.icon,
        appBundleId: packageJson.appId,
        appCategoryType: 'public.app-category.graphics-design',
        extendInfo: './src/public/Info.plist',
        ...(process.argv.includes('--dry-run') || process.argv.includes('--no-sign')
            ? {}
            : {
                  osxSign: {},
                  osxNotarize: {
                      tool: 'notarytool',
                      appleId: process.env.APPLE_ID,
                      appleIdPassword: process.env.APPLE_PASSWORD,
                      teamId: process.env.APPLE_TEAM_ID,
                  },
              }),
    },
    publishers: [
        {
            name: '@electron-forge/publisher-github',
            config: {
                repository: {
                    owner: packageJson.author.name,
                    name: packageJson.productName,
                },
            },
            draft: true,
            prerelease: true,
        },
    ],
    makers: [
        {
            name: '@electron-forge/maker-squirrel',
            config: {
                name: packageJson.productName,
                iconUrl:
                    'https://github.com/chaiNNer-org/chaiNNer/blob/main/src/public/icons/win/icon.ico',
                setupIcon: './src/public/icons/win/icon.ico',
                loadingGif: './src/public/icons/win/installing_loop.gif',
            },
        },
        {
            name: '@electron-forge/maker-zip',
            platforms: ['linux', 'win32'],
        },
        {
            name: '@electron-forge/maker-dmg',
            config: {
                format: 'ULMO',
                background: './src/public/dmg-background.png',
                icon: './src/public/icons/mac/icon.icns',
                additionalDMGOptions: {
                    window: { size: { width: 660, height: 500 } },
                    filesystem: 'APFS',
                    ...(process.argv.includes('--dry-run') || process.argv.includes('--no-sign')
                        ? {}
                        : {
                              'code-sign': {
                                  'signing-identity': process.env.APPLE_SIGNING_ID,
                                  identifier: packageJson.appId,
                              },
                          }),
                },
            },
        },
        {
            name: '@electron-forge/maker-deb',
            config: {
                options: {
                    ...makerOptions,
                    section: 'graphics',
                },
            },
        },
        {
            name: '@electron-forge/maker-rpm',
            config: {
                options: { ...makerOptions, compressionLevel: 9 },
            },
        },
    ],
    plugins: [
        {
            name: '@electron-forge/plugin-vite',
            config: {
                // `build` can specify multiple entry builds, which can be
                // Main process, Preload scripts, Worker process, etc.
                build: [
                    {
                        // `entry` is an alias for `build.lib.entry`
                        // in the corresponding file of `config`.
                        entry: 'src/main/main.ts',
                        config: 'vite/main.config.ts',
                    },
                    {
                        entry: 'src/main/preload.ts',
                        config: 'vite/preload.config.ts',
                        target: 'preload',
                    },
                ],
                renderer: [
                    {
                        name: 'main_window',
                        config: 'vite/renderer.config.ts',
                    },
                ],
            },
        },
    ],
    hooks: {
        prePackage: async (forgeConfig, platform) => {
            // VOSR's TensorRT plugin is an optional target of the native build (Windows);
            // a release without it has no VOSR TensorRT backends (MAINTAINING.md, section 8).
            const vosrPlugin = path.join(__dirname, 'backend/src/nodes/impl/pytorch/vosr2_qkv.dll');
            if (
                platform === 'win32' &&
                process.env.CHAINNER_C_PACKAGE_WITHOUT_VOSR_PLUGIN !== '1' &&
                !nodeFs.existsSync(vosrPlugin)
            ) {
                throw new Error(
                    `${vosrPlugin} is missing: VOSR's TensorRT plugin was not built, and a ` +
                        'package without it has no VOSR TensorRT backends. Build it ' +
                        '(native/README.md, Build), or set ' +
                        'CHAINNER_C_PACKAGE_WITHOUT_VOSR_PLUGIN=1 to package without it.'
                );
            }
            if (platform === 'linux') {
                const backendSrc = path.join(__dirname, 'backend/src');
                const required =
                    process.env.CHAINNER_C_PACKAGE_WITHOUT_VOSR_PLUGIN === '1'
                        ? linuxNativeBinaries
                        : [...linuxNativeBinaries, linuxVosrPlugin];
                const missing = required.filter(
                    (binary) => !nodeFs.existsSync(path.join(backendSrc, binary))
                );
                if (missing.length > 0) {
                    throw new Error(
                        `${backendSrc} lacks ${missing.join(', ')}: the native layer was not ` +
                            'built for Linux (native/README.md, Build). A package without ' +
                            "VOSR's TensorRT plugin has no VOSR TensorRT backends: set " +
                            'CHAINNER_C_PACKAGE_WITHOUT_VOSR_PLUGIN=1 to package without it.'
                    );
                }
            }
            // delete all .pyc files from backend folder, recursively
            const backendPath = path.join(__dirname, 'backend/src');
            await deletePycFiles(backendPath);
        },
        postPackage: async (forgeConfig, { platform, outputPaths }) => {
            if (platform === 'linux') {
                await Promise.all(
                    outputPaths.map(async (outputPath) => {
                        // texconv.exe, which reads and writes DDS, is a Windows program; the DDS
                        // nodes report that on other systems
                        await fs.rm(path.join(outputPath, 'resources/src/texconv'), {
                            recursive: true,
                            force: true,
                        });
                        // The packager makes the folder readable by its owner only, and a zip
                        // carries that to the folder it is unzipped to
                        await fs.chmod(outputPath, 0o755);
                    })
                );
            }
        },
        postMake: async (forgeConfig, makeResults) => {
            const justArtifacts = makeResults.map((m) => m.artifacts).reduce((a, b) => a.concat(b));
            const zipArtifact = justArtifacts.find((a) => a.endsWith('.zip'));
            if (zipArtifact) {
                // Add an empty `portable` file to the zip
                const zip = new AdmZip(zipArtifact);
                switch (process.platform) {
                    case 'win32':
                        zip.addFile('portable', Buffer.alloc(0));
                        // cmd and PowerShell neither wait for a GUI program like chaiNNer.exe nor
                        // get its exit code, but they do for a batch file that starts it (CLI mode)
                        zip.addFile(
                            `${packageJson.productName}-cli.cmd`,
                            Buffer.from(`@"%~dp0${packageJson.productName}.exe" %*\r\n`)
                        );
                        break;
                    case 'linux':
                        zip.addFile('chaiNNer-linux-x64/portable', Buffer.alloc(0));
                        // The launcher starts chainner (CLI mode too) and decides whether
                        // Chromium's sandbox can start; a file added here needs its mode set
                        zip.addFile(
                            `chaiNNer-linux-x64/${packageJson.productName}.sh`,
                            nodeFs.readFileSync(
                                path.join(__dirname, 'src/public/linux/chaiNNer.sh')
                            ),
                            '',
                            0o755
                        );
                        break;
                    default:
                        break;
                }
                zip.writeZip(zipArtifact);
            }
        },
    },
};

module.exports = config;
