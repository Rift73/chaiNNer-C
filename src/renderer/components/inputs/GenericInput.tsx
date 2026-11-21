import { QuestionIcon } from '@chakra-ui/icons';
import { Box, Center, HStack, Text } from '@chakra-ui/react';
import { memo, useState } from 'react';
import { useContext } from 'use-context-selector';
import { InputContext } from '../../contexts/InputContext';
import { IMAGE_NAME_MAX_LENGTH, imageLabel } from '../../helpers/imageCompare';
import { HintTooltip } from '../HintTooltip';
import { RenameInput } from '../RenameInput';
import { TypeTags } from '../TypeTag';
import { WithoutLabel } from './InputContainer';
import { InputProps } from './props';

export const GenericInput = memo(
    ({ input, definitionType, sequenceType, isLocked, rename }: InputProps<'generic'>) => {
        const { label, optional, hint, description } = input;

        const { conditionallyInactive } = useContext(InputContext);

        const [isRenaming, setIsRenaming] = useState(false);
        const shownLabel = rename ? imageLabel(label, rename.name) : label;
        const canRename = rename !== undefined && !isLocked;

        return (
            <WithoutLabel>
                <HintTooltip description={hint ? description : undefined}>
                    <Box
                        display="flex"
                        flexDirection="row"
                        minW={rename ? 0 : undefined}
                    >
                        <HStack
                            m={0}
                            minW={rename ? 0 : undefined}
                            p={0}
                            spacing={0}
                        >
                            {isRenaming && rename ? (
                                <RenameInput
                                    maxLength={IMAGE_NAME_MAX_LENGTH}
                                    name={shownLabel}
                                    onCancel={() => setIsRenaming(false)}
                                    onChangeName={(newName) => {
                                        setIsRenaming(false);
                                        rename.setName(newName);
                                    }}
                                />
                            ) : (
                                <Text
                                    noOfLines={rename ? 1 : undefined}
                                    opacity={conditionallyInactive ? 0.7 : undefined}
                                    textDecoration={
                                        conditionallyInactive ? 'line-through' : undefined
                                    }
                                    title={
                                        canRename
                                            ? `${shownLabel} · Rename (double click)`
                                            : undefined
                                    }
                                    wordBreak={rename ? 'break-all' : undefined}
                                    onDoubleClick={
                                        canRename ? () => setIsRenaming(true) : undefined
                                    }
                                >
                                    {shownLabel}
                                </Text>
                            )}
                            {hint && (
                                <Center
                                    h="auto"
                                    m={0}
                                    p={0}
                                >
                                    <QuestionIcon
                                        boxSize={3}
                                        ml={1}
                                    />
                                </Center>
                            )}
                            <Center>
                                <TypeTags
                                    isOptional={optional}
                                    sequenceType={sequenceType}
                                    type={definitionType}
                                />
                            </Center>
                        </HStack>
                    </Box>
                </HintTooltip>
            </WithoutLabel>
        );
    }
);
