import { useEffect, useState } from 'react';
import { useReactFlow } from 'reactflow';
import { useContextSelector } from 'use-context-selector';
import { EdgeData, NodeData } from '../../common/common-types';
import { EMPTY_ARRAY } from '../../common/util';
import { BackendContext } from '../contexts/BackendContext';
import { GlobalVolatileContext } from '../contexts/GlobalNodeState';
import { UpstreamSeek, getUpstreamSeeks, hasUpstreamFullLoop } from '../helpers/videoSeek';

export interface UpstreamSeeks {
    readonly seeks: readonly UpstreamSeek[];
    /** Whether a loop source before the node runs all its items (hasUpstreamFullLoop). */
    readonly hasFullLoop: boolean;
}
const NONE: UpstreamSeeks = { seeks: EMPTY_ARRAY, hasFullLoop: false };

const seeksKey = ({ seeks, hasFullLoop }: UpstreamSeeks): string =>
    seeks
        .map((s) => {
            const video = s.video ? `${s.video.frameCount}:${s.video.fps}` : '-';
            return `${s.nodeId}:${s.input.id}:${s.frame}:${video}`;
        })
        .concat(String(hasFullLoop))
        .join(';');

/**
 * The seek bars before node `nodeId` (getUpstreamSeeks), kept up to date with the chain, the
 * bars' frames and their videos, and whether a loop source before it runs all its items; none
 * while `isEnabled` is false, and then the node follows no change of the chain.
 */
export const useUpstreamSeeks = (nodeId: string, isEnabled: boolean): UpstreamSeeks => {
    const { getNodes, getEdges } = useReactFlow<NodeData, EdgeData>();
    const schemata = useContextSelector(BackendContext, (c) => c.schemata);
    const nodeChanges = useContextSelector(GlobalVolatileContext, (c) =>
        isEnabled ? c.nodeChanges : undefined
    );
    const edgeChanges = useContextSelector(GlobalVolatileContext, (c) =>
        isEnabled ? c.edgeChanges : undefined
    );
    const outputDataMap = useContextSelector(GlobalVolatileContext, (c) =>
        isEnabled ? c.outputDataMap : undefined
    );

    const [upstream, setUpstream] = useState<UpstreamSeeks>(NONE);
    useEffect(() => {
        let next = NONE;
        if (outputDataMap !== undefined) {
            const nodes = getNodes().map((n) => n.data);
            const edges = getEdges();
            const getSchema = (schemaId: NodeData['schemaId']) => schemata.get(schemaId);
            next = {
                seeks: getUpstreamSeeks(
                    nodeId,
                    nodes,
                    edges,
                    getSchema,
                    (id) => outputDataMap.get(id)?.types
                ),
                hasFullLoop: hasUpstreamFullLoop(nodeId, nodes, edges, getSchema),
            };
        }
        setUpstream((prev) => (seeksKey(prev) === seeksKey(next) ? prev : next));
    }, [nodeId, nodeChanges, edgeChanges, outputDataMap, schemata, getNodes, getEdges]);
    return upstream;
};
