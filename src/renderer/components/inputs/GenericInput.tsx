import { QuestionIcon } from '@chakra-ui/icons';
import { Box, Center, HStack, Text } from '@chakra-ui/react';
import { memo } from 'react';
import { useContext } from 'use-context-selector';
import { InputContext } from '../../contexts/InputContext';
import { HintTooltip } from '../HintTooltip';
import { TypeTags } from '../TypeTag';
import { WithoutLabel } from './InputContainer';
import { InputProps } from './props';

export const GenericInput = memo(
    ({ input, definitionType, sequenceType }: InputProps<'generic'>) => {
        const { label, optional, hint, description } = input;

        const { conditionallyInactive } = useContext(InputContext);

        return (
            <WithoutLabel>
                <HintTooltip description={hint ? description : undefined}>
                    <Box
                        display="flex"
                        flexDirection="row"
                    >
                        <HStack
                            m={0}
                            p={0}
                            spacing={0}
                        >
                            <Text
                                opacity={conditionallyInactive ? 0.7 : undefined}
                                textDecoration={conditionallyInactive ? 'line-through' : undefined}
                            >
                                {label}
                            </Text>
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
