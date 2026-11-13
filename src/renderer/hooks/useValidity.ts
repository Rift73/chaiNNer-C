import { useEffect, useMemo, useState } from 'react';
import { useReactFlow } from 'reactflow';
import { useContext, useContextSelector } from 'use-context-selector';
import { EdgeData, InputData, NodeData, NodeSchema } from '../../common/common-types';
import { checkFeatures } from '../../common/nodes/checkFeatures';
import { checkNodeValidity, checkRequiredInputs } from '../../common/nodes/checkNodeValidity';
import { getConnectedInputs } from '../../common/nodes/connectedInputs';
import { VALID, Validity, bothValid, invalid } from '../../common/Validity';
import { BackendContext } from '../contexts/BackendContext';
import { GlobalVolatileContext } from '../contexts/GlobalNodeState';

const STARTING_VALIDITY: Validity = invalid('Validating nodes...');

export interface UseValidity {
    validity: Validity;
}

export const useValidity = (id: string, schema: NodeSchema, inputData: InputData): UseValidity => {
    const edgeChanges = useContextSelector(GlobalVolatileContext, (c) => c.edgeChanges);
    const functionInstance = useContextSelector(GlobalVolatileContext, (c) =>
        c.typeState.functions.get(id)
    );
    const chainLineage = useContextSelector(GlobalVolatileContext, (c) => c.chainLineage);
    const { getEdges } = useReactFlow<NodeData, EdgeData>();

    const alwaysValid = schema.inputs.length === 0;

    const guaranteedMissingInputs = useMemo(
        (): Validity => checkRequiredInputs(schema, inputData),
        [schema, inputData]
    );

    const [fullValidity, setFullValidity] = useState<Validity>(
        alwaysValid ? VALID : STARTING_VALIDITY
    );
    useEffect(() => {
        if (!alwaysValid) {
            setFullValidity(
                checkNodeValidity({
                    schema,
                    inputData,
                    connectedInputs: getConnectedInputs(id, getEdges()),
                    functionInstance,
                    chainLineage,
                    nodeId: id,
                })
            );
        }
    }, [alwaysValid, id, schema, inputData, edgeChanges, getEdges, functionInstance, chainLineage]);

    // `checkNodeValidity` runs after a delay (it needs `getEdges`), so the full validity can be
    // stale, while other code expects a valid node to be runnable. A conservative validity is also
    // computed at once: invalid only when the node is surely invalid. This narrows the gap, it does
    // not close it.
    const validity =
        fullValidity.isValid && !guaranteedMissingInputs.isValid
            ? guaranteedMissingInputs
            : fullValidity;

    const { features, featureStates } = useContext(BackendContext);
    const featureValidity = useMemo(() => {
        return checkFeatures(schema.features, features, featureStates);
    }, [schema, features, featureStates]);

    return { validity: bothValid(featureValidity, validity) };
};
