import { expect, test } from 'vitest';
import { getConnectionState } from '../../src/renderer/helpers/connectionState';

test('the editor stays when the live node list fails and it has a list', () => {
    // The cached list (or the last one) is shown: unmounting the editor would lose its unsaved
    // changes. The failure is reported by an alert.
    expect(getConnectionState(true, true)).toBe('connected');
});

test('without a node list, the editor waits for one, and a failure ends the wait', () => {
    expect(getConnectionState(false, false)).toBe('connecting');
    expect(getConnectionState(true, false)).toBe('failed');
    expect(getConnectionState(false, true)).toBe('connected');
});
