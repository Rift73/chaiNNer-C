import { Node } from 'reactflow';
import { NodeData } from '../../common/common-types';
import { SchemaMap } from '../../common/SchemaMap';

export type ChainProgress = ReadonlyMap<string, ChainNodeProgress>;

export interface ChainNodeProgress {
    readonly progress: number;
    readonly weight: number;
}

export const getTotalProgress = (chainProgress: ChainProgress): number => {
    if (chainProgress.size === 0) return 0;
    let totalProgress = 0;
    let totalWeight = 0;
    for (const { progress, weight } of chainProgress.values()) {
        totalProgress += progress * weight;
        totalWeight += weight;
    }
    return totalProgress / totalWeight;
};

interface NodeProgressValues {
    readonly progress: number;
    readonly index: number;
    readonly total: number;
}

/** A node's own progress (NodeContext.set_progress), as opposed to a generator's items: it has none. */
export const isOwnProgress = ({ total }: NodeProgressValues): boolean => total === 0;

export const withNodeProgress = (
    chainProgress: ChainProgress,
    nodeId: string,
    progress: number
): ChainProgress => {
    const node = chainProgress.get(nodeId);
    if (!node) return chainProgress;
    return new Map(chainProgress).set(nodeId, { progress, weight: node.weight });
};

/**
 * The nodes' progress once a node finished. Its own progress ends at 100%: the backend's last
 * update of a pass, at most ten a second and never its 100%, is older than the finish.
 */
export const withNodeFinished = <T extends NodeProgressValues>(
    nodeProgress: Readonly<Record<string, T | undefined>>,
    nodeId: string
): Readonly<Record<string, T | undefined>> => {
    const progress = nodeProgress[nodeId];
    if (!progress || !isOwnProgress(progress)) return nodeProgress;
    return { ...nodeProgress, [nodeId]: { ...progress, progress: 1 } };
};

export const nodeProgressLabel = (values: NodeProgressValues): string => {
    const percent = `${(values.progress * 100).toFixed(0)}%`;
    return isOwnProgress(values) ? percent : `${values.index}/${values.total} (${percent})`;
};

export const getInitialChainProgress = (
    nodes: readonly Node<NodeData>[],
    schemata: SchemaMap
): ChainProgress => {
    const progress = new Map<string, ChainNodeProgress>();

    // TODO: implement proper weights based on iterated nodes

    for (const node of nodes) {
        let weight = 1;
        if (schemata.get(node.data.schemaId).kind === 'generator') {
            // more weight for iterators
            weight = 4;
        }
        progress.set(node.id, { progress: 0, weight });
    }

    return progress;
};
