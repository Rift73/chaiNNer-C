import { Type } from '@chainner/navi';
import { ImageCompare, NodeSchema, Output, OutputId, Size } from '../../../common/common-types';

export interface UseOutputData<T> {
    /** The current output data. Current here means most recent + up to date (= same input hash). */
    readonly current: T | undefined;
    /** The most recent output data. */
    readonly last: T | undefined;
    /** Whether the most recent output data ({@link last}) is not the current output data ({@link current}). */
    readonly stale: boolean;
}

export interface CompareOption {
    readonly id: OutputId;
    readonly label: string;
}

/** The images of a node's one viewer for its several large image outputs. */
export interface CompareImages {
    readonly options: readonly CompareOption[];
    readonly choice: ImageCompare | undefined;
    readonly setChoice: (choice: ImageCompare) => void;
}

export interface OutputProps {
    readonly output: Output;
    readonly id: string;
    readonly schema: NodeSchema;
    readonly definitionType: Type;
    readonly type: Type;
    readonly useOutputData: <T>(outputId: OutputId) => UseOutputData<T>;
    readonly animated: boolean;
    readonly size: Readonly<Size> | undefined;
    readonly setSize: (size: Readonly<Size>) => void;
    readonly sequenceType?: Type;
    /** The node's seek bar, for a large image output's maximized view. */
    readonly seekBar?: JSX.Element;
    /** Set for the viewer of a node with several large image outputs. */
    readonly compare?: CompareImages;
}
