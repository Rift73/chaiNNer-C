import { Condition, InputId, NodeSchema } from '../common-types';
import { getGroupStacks } from './groupStacks';

/** The conditions of all conditional groups the input is in, at any depth. */
export const getInputCondition = (schema: NodeSchema, inputId: InputId): Condition | undefined => {
    const conditions = getGroupStacks(schema).inputConditions.get(inputId);
    return conditions && { kind: 'and', items: [...conditions] };
};
