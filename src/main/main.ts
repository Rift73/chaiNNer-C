import electronLog from 'electron-log/main';
import { app } from 'electron/main';
import path from 'path';
import './i18n';
import { LEVEL_NAME, log } from '../common/log';
import { parseArgs } from './arguments';
import { createCli } from './cli/create';
import { createGuiApp } from './gui/create';
import { getLogsFolder, getRootDir } from './platform';
import { handleSquirrel } from './squirrel';
import { checkUnfinishedUpdate, removeUnpackedReleases, showUpdateAtStart } from './updater';

const startApp = () => {
    const args = parseArgs(process.argv.slice(app.isPackaged ? 1 : 2));

    electronLog.initialize();
    electronLog.transports.file.resolvePathFn = (variables) =>
        path.join(getLogsFolder(), variables.fileName!);
    electronLog.transports.file.level = 'info';
    // A long batch outgrows the 1 MB default and loses its first lines to the rotation.
    electronLog.transports.file.maxSize = 64 * 1024 * 1024;
    electronLog.transports.console.level = 'debug';

    log.addTransport({
        log: ({ level, message, additional }) => {
            electronLog[LEVEL_NAME[level]](message, ...additional);
        },
    });

    // Writing to a terminal or pipe that has gone away fails (EIO, EPIPE). Unhandled, that error
    // becomes an error dialog (GUI) or ends the run (CLI). Stop logging to the console instead;
    // the log file keeps everything, including this error.
    const stopConsoleLogging = (error: Error) => {
        if (electronLog.transports.console.level !== false) {
            electronLog.transports.console.level = false;
            log.warn('Stopped logging to the console after a write error.', error);
        }
    };
    process.stdout.on('error', stopConsoleLogging);
    process.stderr.on('error', stopConsoleLogging);

    process.env.ELECTRON_DISABLE_SECURITY_WARNINGS = 'true';

    app.setPath('userData', getRootDir());

    // On macOS, we need to store the file-path when chaiNNer got started via a double
    // click on a .chn file. This listener gets remove later on.
    app.on('open-file', (event, filePath) => {
        event.preventDefault();
        globalThis.startupFile = filePath;
    });

    app.on('quit', removeUnpackedReleases);

    // A command-line run the update kept from running ends with exit code 1, as a run that failed
    // does (cli/create.ts); a window session just ends.
    const endStart = () => {
        if (args.command === 'open') {
            app.quit();
        } else {
            app.exit(1);
        }
    };

    // An update that did not finish keeps the app from starting on a half-swapped install.
    checkUnfinishedUpdate()
        .then(async (message) => {
            if (message) {
                if (await showUpdateAtStart(message)) {
                    // a release kept by the user: the next start finds no update left. Only a
                    // window session starts again; a command-line run is not repeated on its own.
                    if (args.command === 'open') {
                        app.relaunch();
                    } else {
                        log.error('The release in the install was kept. Run the command again.');
                    }
                } else {
                    await message.afterwards?.();
                }
                endStart();
            } else if (args.command === 'open') {
                createGuiApp(args);
            } else {
                // Only a run needs the type engine and its regex WebAssembly: a window launch does not load them.
                createCli(async () => {
                    const { runChainInCli } = await import('./cli/run');
                    await runChainInCli(args);
                });
            }
        })
        .catch(async (error) => {
            log.error('Could not finish the last update.', error);
            await showUpdateAtStart({ title: 'The update did not finish', text: String(error) });
            endStart();
        });
};

if (handleSquirrel()) {
    app.quit();
} else {
    startApp();
}
