import { NeverType, Type, evaluate } from '@chainner/navi';
import { Fragment, memo, useCallback, useEffect, useMemo } from 'react';
import { useContext, useContextSelector } from 'use-context-selector';
import { OutputId, OutputKind, Size } from '../../../common/common-types';
import { log } from '../../../common/log';
import { getGroupStacks } from '../../../common/nodes/groupStacks';
import { getChainnerScope } from '../../../common/types/chainner-scope';
import { ExpressionJson, fromJson } from '../../../common/types/json';
import { BackendContext } from '../../contexts/BackendContext';
import { GlobalContext, GlobalVolatileContext } from '../../contexts/GlobalNodeState';
import { WithInputContext } from '../../contexts/InputContext';
import { NodeState } from '../../helpers/nodeState';
import { getSeekUnderPreview } from '../../helpers/videoSeek';
import { useAutomaticFeatures } from '../../hooks/useAutomaticFeatures';
import { useIsCollapsedNode } from '../../hooks/useIsCollapsedNode';
import { SchemaInput } from '../inputs/SchemaInput';
import { SeekBarLayoutContext } from '../inputs/VideoSeekInput';
import { GenericOutput } from '../outputs/GenericOutput';
import { LargeImageOutput } from '../outputs/LargeImageOutput';
import { OutputContainer } from '../outputs/OutputContainer';
import { OutputProps, UseOutputData } from '../outputs/props';
import { TaggedOutput } from '../outputs/TaggedOutput';

const OutputComponents: Readonly<
    Record<OutputKind, React.MemoExoticComponent<(props: OutputProps) => JSX.Element>>
> = {
    'large-image': LargeImageOutput,
    tagged: TaggedOutput,
    generic: GenericOutput,
};
const OutputIsGeneric: Readonly<Record<OutputKind, boolean>> = {
    'large-image': false,
    tagged: false,
    generic: true,
};

const NO_OUTPUT_DATA: UseOutputData<never> = { current: undefined, last: undefined, stale: false };

const evalExpression = (expression: ExpressionJson | null | undefined): Type | undefined => {
    if (expression == null) return undefined;
    try {
        return evaluate(fromJson(expression), getChainnerScope());
    } catch (error) {
        log.error(error);
    }
};

interface NodeOutputProps {
    nodeState: NodeState;
    animated: boolean;
}

export const NodeOutputs = memo(({ nodeState, animated }: NodeOutputProps) => {
    const {
        id,
        schema,
        schemaId,
        outputHeight,
        setOutputHeight,
        nodeWidth,
        setWidth,
        iteratedOutputs,
    } = nodeState;

    const { functionDefinitions } = useContext(BackendContext);
    const { setManualOutputType, setManualSequenceOutputType } = useContext(GlobalContext);
    const outputDataEntry = useContextSelector(GlobalVolatileContext, (c) =>
        c.outputDataMap.get(id)
    );
    const inputHash = useContextSelector(GlobalVolatileContext, (c) => c.inputHashes.get(id));
    const stale = inputHash !== outputDataEntry?.inputHash;

    const useOutputData = useCallback(
        // eslint-disable-next-line prefer-arrow-functions/prefer-arrow-functions, func-names
        function <T>(outputId: OutputId): UseOutputData<T> {
            if (outputDataEntry) {
                const last = outputDataEntry.data?.[outputId] as T | undefined;
                if (last !== undefined) {
                    return { current: stale ? undefined : last, last, stale };
                }
            }
            return NO_OUTPUT_DATA;
        },
        [outputDataEntry, stale]
    );

    const currentTypes = stale ? undefined : outputDataEntry?.types;
    const currentSequenceTypes = stale ? undefined : outputDataEntry?.sequenceTypes;

    const { appliesRunTypes } = useAutomaticFeatures(id, schemaId);

    useEffect(() => {
        if (appliesRunTypes) {
            for (const output of schema.outputs) {
                const type = evalExpression(currentTypes?.[output.id]);
                setManualOutputType(id, output.id, type);
            }
            for (const iterOutput of schema.iteratorOutputs) {
                const type = evalExpression(currentSequenceTypes?.[iterOutput.id]);
                setManualSequenceOutputType(id, iterOutput.id, type);
            }
        }
    }, [
        id,
        currentTypes,
        currentSequenceTypes,
        schema,
        setManualOutputType,
        setManualSequenceOutputType,
        appliesRunTypes,
    ]);

    // a seek bar belongs with the picture it seeks: under the Preview, and in its maximized view
    const seek = getSeekUnderPreview(schema);
    const seekBar = seek && (
        <SchemaInput
            input={seek.input}
            nodeState={nodeState}
        />
    );
    // Frames, Start and End go with the bar, under the same conditions as among the inputs: shown
    // when their groups' conditions hold, and inactive but shown while connected otherwise.
    const seekLayout = useMemo(() => {
        const companions = seek?.companions.map((input) => {
            const conditions = getGroupStacks(schema).inputConditions.get(input.id) ?? [];
            const element = (
                <SchemaInput
                    input={input}
                    key={input.id}
                    nodeState={nodeState}
                />
            );
            if (conditions.every(nodeState.testCondition)) return element;
            if (!nodeState.connectedInputs.has(input.id)) return null;
            return (
                <WithInputContext
                    conditionallyInactive
                    key={input.id}
                >
                    {element}
                </WithInputContext>
            );
        });
        return { isWide: false, companions };
    }, [seek, schema, nodeState]);

    const isCollapsed = useIsCollapsedNode();
    if (isCollapsed) {
        // just need the effect for collapsed nodes, not the elements
        return null;
    }

    const functionDef = functionDefinitions.get(schemaId);
    const functions = functionDef?.outputDefaults;
    return (
        <SeekBarLayoutContext.Provider value={seekLayout}>
            {schema.outputs.map((output) => {
                if (schema.inputs.some((i) => i.fused?.outputId === output.id)) {
                    return null;
                }

                const definitionType = functions?.get(output.id) ?? NeverType.instance;

                const type = nodeState.type.instance?.outputs.get(output.id);
                const outputLength = nodeState.type.instance?.outputSequence.get(output.id);

                const size =
                    outputHeight?.[output.id] && nodeWidth
                        ? { height: outputHeight[output.id], width: nodeWidth }
                        : undefined;
                const setSize = (newSize: Readonly<Size>) => {
                    setOutputHeight(output.id, newSize.height);
                    setWidth(newSize.width);
                };

                const OutputType = OutputComponents[output.kind];
                const isSeekOutput = seek?.outputId === output.id;
                return (
                    <Fragment key={`${id}-${output.id}`}>
                        <OutputContainer
                            definitionType={definitionType}
                            generic={OutputIsGeneric[output.kind]}
                            id={id}
                            isConnected={nodeState.connectedOutputs.has(output.id)}
                            isIterated={iteratedOutputs.has(output.id)}
                            output={output}
                            sequenceType={outputLength}
                            type={type}
                        >
                            <OutputType
                                animated={animated}
                                definitionType={definitionType}
                                id={id}
                                output={output}
                                schema={nodeState.schema}
                                seekBar={isSeekOutput ? seekBar : undefined}
                                sequenceType={outputLength}
                                setSize={setSize}
                                size={size}
                                type={type ?? NeverType.instance}
                                useOutputData={useOutputData}
                            />
                        </OutputContainer>
                        {isSeekOutput && seekBar}
                    </Fragment>
                );
            })}
        </SeekBarLayoutContext.Provider>
    );
});
