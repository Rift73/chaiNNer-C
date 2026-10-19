import { expect, test } from 'vitest';
import { EditHistory } from '../../src/renderer/helpers/editHistory';

interface Step {
    readonly key: string;
    readonly label: string;
}
const step = (key: string, label = key): Step => ({ key, label });

test('a commit equal to the current step is not a new step', () => {
    const history = EditHistory.create(step('a'), 10).commit(step('b'));
    const same = history.commit(step('b', 'b with other objects'));

    expect(same).toBe(history);
    expect(same.current.label).toBe('b');
    expect(same.undo().current.key).toBe('a');
    expect(same.undo().undo().current.key).toBe('a');
});

test('an equal commit after an undo keeps the redo steps', () => {
    const history = EditHistory.create(step('a'), 10).commit(step('b')).commit(step('c'));
    const undone = history.undo();

    const afterEqual = undone.commit(step('b'));
    expect(afterEqual.redo().current.key).toBe('c');

    // a different step still replaces the redo steps
    const afterNew = undone.commit(step('d'));
    expect(afterNew.current.key).toBe('d');
    expect(afterNew.redo().current.key).toBe('d');
    expect(afterNew.undo().current.key).toBe('b');
});

test('the history keeps at most maxLength steps', () => {
    let history = EditHistory.create(step('0'), 3);
    for (const key of ['1', '2', '3', '4']) {
        history = history.commit(step(key));
    }
    const keys = [history.current.key];
    for (let i = 0; i < 5; i += 1) {
        history = history.undo();
        keys.push(history.current.key);
    }
    expect(keys).toEqual(['4', '3', '2', '2', '2', '2']);
});
