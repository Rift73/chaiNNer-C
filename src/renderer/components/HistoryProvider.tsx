import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { Edge, Node, useReactFlow } from 'reactflow';
import { useContext, useContextSelector } from 'use-context-selector';
import { EdgeData, NodeData } from '../../common/common-types';
import { chainSnapshot } from '../../common/nodes/chainSnapshot';
import { noop } from '../../common/util';
import { GlobalContext, GlobalVolatileContext } from '../contexts/GlobalNodeState';
import { EditHistory } from '../helpers/editHistory';
import { useHotkeys } from '../hooks/useHotkeys';
import { useIpcRendererListener } from '../hooks/useIpcRendererListener';

interface HistoryState {
    readonly nodes: Node<NodeData>[];
    readonly edges: Edge<EdgeData>[];
    /** The `chainSnapshot` of the nodes and edges: steps with the same chain are one step. */
    readonly key: string;
}
const historyState = (nodes: Node<NodeData>[], edges: Edge<EdgeData>[]): HistoryState => ({
    nodes,
    edges,
    key: chainSnapshot(nodes, edges),
});

export const HistoryProvider = memo(
    ({ children }: React.PropsWithChildren<unknown>): JSX.Element => {
        const changeId = useContextSelector(
            GlobalVolatileContext,
            (c) => `${c.nodeChanges},${c.edgeChanges}`
        );
        const chainLoads = useContextSelector(GlobalVolatileContext, (c) => c.chainLoads);
        const { changeNodes, changeEdges, flushInputChanges } = useContext(GlobalContext);
        const { getNodes, getEdges } = useReactFlow();

        // eslint-disable-next-line react/hook-use-state
        const [initialValue] = useState(() => {
            return EditHistory.create(historyState(getNodes(), getEdges()), 100);
        });
        const historyRef = useRef(initialValue);
        const commitTimerRef = useRef<ReturnType<typeof setTimeout>>();
        const committedChainLoads = useRef(chainLoads);

        const [selfUpdate, setSelfUpdate] = useState(false);
        useEffect(() => {
            if (!selfUpdate) return noop;

            const id = setTimeout(() => {
                setSelfUpdate(false);
            }, 50);
            return () => clearTimeout(id);
        }, [selfUpdate, setSelfUpdate]);

        const apply = useCallback(
            ({ nodes, edges }: HistoryState) => {
                flushSync(() => {
                    setSelfUpdate(true);
                    changeNodes(nodes);
                    changeEdges(edges);
                });
            },
            [setSelfUpdate, changeNodes, changeEdges]
        );

        const commitCurrent = useCallback(() => {
            clearTimeout(commitTimerRef.current);
            const state = historyState(
                getNodes().map((n) => ({ ...n, selected: false })),
                getEdges()
            );
            if (committedChainLoads.current === chainLoads) {
                historyRef.current = historyRef.current.commit(state);
            } else {
                // An opened chain starts a new history: its file's path is the save path now, so
                // a step back into the chain before it would be saved over that file.
                committedChainLoads.current = chainLoads;
                historyRef.current = EditHistory.create(state, historyRef.current.maxLength);
            }
        }, [getNodes, getEdges, chainLoads]);

        // commit to history
        useEffect(() => {
            if (selfUpdate) return noop;

            const id = setTimeout(commitCurrent, 250);
            commitTimerRef.current = id;
            return () => clearTimeout(id);
        }, [changeId, commitCurrent, selfUpdate]);

        const undo = useCallback(() => {
            flushInputChanges();
            commitCurrent();
            historyRef.current = historyRef.current.undo();
            apply(historyRef.current.current);
        }, [apply, commitCurrent, flushInputChanges]);
        const redo = useCallback(() => {
            flushInputChanges();
            commitCurrent();
            historyRef.current = historyRef.current.redo();
            apply(historyRef.current.current);
        }, [apply, commitCurrent, flushInputChanges]);

        useIpcRendererListener('history-undo', undo);
        useIpcRendererListener('history-redo', redo);
        useHotkeys('ctrl+z, cmd+z', undo);
        useHotkeys('ctrl+y, cmd+y, ctrl+shift+z, cmd+shift+z', redo);

        // eslint-disable-next-line react/jsx-no-useless-fragment
        return <>{children}</>;
    }
);
