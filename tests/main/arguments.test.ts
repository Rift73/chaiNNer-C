import { expect, test } from 'vitest';
import { parseArgs } from '../../src/main/arguments';

// Chromium switches, such as the Linux launcher's --no-sandbox, are among the app's arguments
test.each([
    [['--no-sandbox'], undefined],
    [['--no-sandbox', 'chain.chn'], 'chain.chn'],
    [['chain.chn', '--no-sandbox'], 'chain.chn'],
    [['--ozone-platform-hint=auto', '--no-sandbox', 'chain.chn'], 'chain.chn'],
])('%j opens %j with --no-sandbox', (args, file) => {
    expect(parseArgs(args)).toMatchObject({ command: 'open', file });
});

test.each([[['--no-sandbox', 'run', 'chain.chn']], [['run', 'chain.chn', '--no-sandbox']]])(
    '%j runs the chain with --no-sandbox',
    (args) => {
        expect(parseArgs(args)).toMatchObject({ command: 'run', file: 'chain.chn' });
    }
);

// Chromium's own switches, which the app does not declare, are passed on to Chromium
test.each([
    [['chain.chn', '--disable-gpu']],
    [['chain.chn', '--disable-gpu', '--force-device-scale-factor=1.5']],
    [['--disable-gpu', 'chain.chn']],
    [['--disable-gpu', '--force-device-scale-factor=1.5', 'chain.chn']],
    [['open', '--disable-gpu', 'chain.chn']],
])('%j opens the chain with switches the app does not know', (args) => {
    expect(parseArgs(args)).toMatchObject({ command: 'open', file: 'chain.chn' });
});

test('two switches the app does not know open the app', () => {
    expect(parseArgs(['--disable-gpu', '--force-device-scale-factor=1.5'])).toMatchObject({
        command: 'open',
    });
});

test.each([[['run', '--disable-gpu', 'chain.chn']], [['run', 'chain.chn', '--disable-gpu']]])(
    '%j runs the chain with a switch the app does not know',
    (args) => {
        expect(parseArgs(args)).toMatchObject({ command: 'run', file: 'chain.chn' });
    }
);
