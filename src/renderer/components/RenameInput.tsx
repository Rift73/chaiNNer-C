import { Input } from '@chakra-ui/react';
import { memo, useEffect, useRef, useState } from 'react';

interface RenameInputProps {
    name: string;
    onChangeName: (name: string) => void;
    onCancel: () => void;
    maxLength?: number;
}

/** A field in place of a name: Enter or leaving the field keeps the new name, Esc cancels. */
export const RenameInput = memo(({ name, onChangeName, onCancel, maxLength }: RenameInputProps) => {
    const [tempName, setTempName] = useState<string>(name);

    useEffect(() => {
        setTempName(name);
    }, [name]);

    const inputRef = useRef<HTMLInputElement>(null);
    useEffect(() => {
        inputRef.current?.select();
    }, []);

    return (
        <Input
            autoFocus
            borderColor="transparent"
            className="nodrag"
            flex={1}
            height="auto"
            htmlSize={0}
            lineHeight="100%"
            maxLength={maxLength}
            my="-0.25rem"
            px={2}
            py="calc(0.25rem - 1px)"
            ref={inputRef}
            size="md"
            textAlign="center"
            value={tempName}
            onBlur={() => {
                onChangeName(tempName);
            }}
            onChange={(e) => {
                setTempName(e.target.value);
            }}
            onKeyDown={(e) => {
                e.stopPropagation();
                if (e.key === 'Enter') {
                    onChangeName(tempName);
                }
                if (e.key === 'Escape') {
                    onCancel();
                }
            }}
        />
    );
});
