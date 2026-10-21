export type ConnectionState = 'connecting' | 'connected' | 'failed';

/**
 * Whether the editor can show: it can with any node list. A backend's list that fails to arrive
 * (or to refresh) is reported by an alert, and the editor keeps the list it has (the one cached at
 * the last start, or the last one), so its unsaved changes stay and can be saved.
 */
export const getConnectionState = (queryFailed: boolean, hasNodeList: boolean): ConnectionState => {
    if (hasNodeList) {
        return 'connected';
    }
    return queryFailed ? 'failed' : 'connecting';
};
