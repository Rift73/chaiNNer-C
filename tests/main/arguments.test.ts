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
