import { useCallback } from 'react';
import { useReactFlow } from 'reactflow';
import { useContextSelector } from 'use-context-selector';
import { EdgeData, NodeData } from '../../common/common-types';
import { RunPart } from '../../common/input-override-common';
import { log } from '../../common/log';
import { BackendContext } from '../contexts/BackendContext';
import { ExecutionContext } from '../contexts/ExecutionContext';
import { GlobalContext } from '../contexts/GlobalNodeState';
import { runThisFrameOverrides } from '../helpers/videoSeek';

/**
 * Starts "Run this frame": the part of the chain `part` names runs, every seek bar's node on one
 * frame (runThisFrameOverrides), those in `pressed` on the frame given for them.
 */
export const useRunThisFrame = (): ((
    pressed: ReadonlyMap<string, number>,
    part: RunPart
) => void) => {
    const { getNodes, getEdges } = useReactFlow<NodeData, EdgeData>();
    const schemata = useContextSelector(BackendContext, (c) => c.schemata);
    // every node's last reported outputs, for the other seek bars' videos, read at the press
    const getOutputData = useContextSelector(GlobalContext, (c) => c.getOutputData);
    const runWithOverrides = useContextSelector(ExecutionContext, (c) => c.runWithOverrides);

    return useCallback(
        (pressed: ReadonlyMap<string, number>, part: RunPart) => {
            const overrides = runThisFrameOverrides(
                getNodes().map((node) => node.data),
                getEdges(),
                (schemaId) => schemata.get(schemaId),
                (id) => getOutputData(id)?.types,
                pressed
            );
            runWithOverrides(overrides, part).catch(log.error);
        },
        [getNodes, getEdges, schemata, getOutputData, runWithOverrides]
    );
};
