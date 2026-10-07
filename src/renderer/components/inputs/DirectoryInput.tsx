import { Type, isStringLiteral } from '@chainner/navi';
import { CloseIcon } from '@chakra-ui/icons';
import {
    Icon,
    Input,
    InputGroup,
    InputLeftElement,
    MenuDivider,
    MenuItem,
    MenuList,
    Tooltip,
} from '@chakra-ui/react';
import { DragEvent, memo } from 'react';
import { useTranslation } from 'react-i18next';
import { BsFolderPlus } from 'react-icons/bs';
import { MdContentCopy, MdFolder } from 'react-icons/md';
import { useContext } from 'use-context-selector';
import { log } from '../../../common/log';

import { getFields, isDirectory } from '../../../common/types/util';
import { AlertBoxContext } from '../../contexts/AlertBoxContext';
import { useContextMenu } from '../../hooks/useContextMenu';
import { useInputRefactor } from '../../hooks/useInputRefactor';
import { useLastDirectory } from '../../hooks/useLastDirectory';
import { ipcRenderer } from '../../safeIpc';
import { AutoLabel } from './InputContainer';
import { InputProps } from './props';

const getDirectoryPath = (type: Type): string | undefined => {
    if (isDirectory(type)) {
        const { path } = getFields(type);
        if (isStringLiteral(path)) {
            return path.value;
        }
    }
    return undefined;
};

export const DirectoryInput = memo(
    ({
        value,
        setValue,
        resetValue,
        isLocked,
        input,
        inputKey,
        isConnected,
        inputType,
        nodeId,
    }: InputProps<'directory', string>) => {
        const { t } = useTranslation();
        const { sendToast } = useContext(AlertBoxContext);

        const { lastDirectory, setLastDirectory } = useLastDirectory(inputKey);

        const onButtonClick = async () => {
            const { canceled, filePaths } = await ipcRenderer.invoke(
                'dir-select',
                value ?? lastDirectory ?? ''
            );
            const path = filePaths[0];
            if (!canceled && path) {
                setValue(path);
                setLastDirectory(path);
            }
        };

        // A directory dropped from the file manager sets the input; a locked or connected input
        // takes no drop, and a file or several items are refused with a toast.
        const onDragOver = (event: DragEvent<HTMLDivElement>) => {
            if (!event.dataTransfer.types.includes('Files')) return;
            event.preventDefault();
            event.stopPropagation();
            // eslint-disable-next-line no-param-reassign
            event.dataTransfer.dropEffect = isLocked || isConnected ? 'none' : 'copy';
        };

        const onDrop = (event: DragEvent<HTMLDivElement>) => {
            if (!event.dataTransfer.types.includes('Files')) return;
            event.preventDefault();
            event.stopPropagation();
            if (isLocked || isConnected) return;

            const { files, items } = event.dataTransfer;
            if (files.length !== 1) {
                sendToast({
                    status: 'error',
                    description: `Only one directory is accepted by ${input.label}.`,
                });
                return;
            }
            const entries = Array.from(items).filter((item) => item.kind === 'file');
            const folder =
                entries.length === 1 && entries[0].webkitGetAsEntry()?.isDirectory === true
                    ? files[0].path
                    : undefined;
            if (folder) {
                setValue(folder);
                setLastDirectory(folder);
            } else {
                sendToast({
                    status: 'error',
                    description: `Drop a directory onto ${input.label}, not a file.`,
                });
            }
        };

        const displayDirectory = isConnected ? getDirectoryPath(inputType) : value;

        const refactor = useInputRefactor(nodeId, input, value, isConnected);

        const menu = useContextMenu(() => (
            <MenuList className="nodrag">
                <MenuItem
                    icon={<BsFolderPlus />}
                    isDisabled={isLocked || isConnected}
                    // eslint-disable-next-line @typescript-eslint/no-misused-promises
                    onClick={onButtonClick}
                >
                    {t('inputs.directory.selectDirectory', 'Select a directory...')}
                </MenuItem>
                <MenuDivider />
                <MenuItem
                    icon={<MdFolder />}
                    isDisabled={!displayDirectory}
                    onClick={() => {
                        if (displayDirectory) {
                            ipcRenderer
                                .invoke('shell-showItemInFolder', displayDirectory)
                                .catch(log.error);
                        }
                    }}
                >
                    {t('inputs.directory.openInFileExplorer', 'Open in File Explorer')}
                </MenuItem>
                <MenuItem
                    icon={<MdContentCopy />}
                    isDisabled={!displayDirectory}
                    onClick={() => {
                        if (displayDirectory) {
                            navigator.clipboard.writeText(displayDirectory).catch(log.error);
                        }
                    }}
                >
                    {t('inputs.directory.copyFullDirectoryPath', 'Copy Full Directory Path')}
                </MenuItem>
                <MenuDivider />
                <MenuItem
                    icon={<CloseIcon />}
                    isDisabled={!value}
                    onClick={resetValue}
                >
                    {t('inputs.directory.clear', 'Clear')}
                </MenuItem>
                {refactor}
            </MenuList>
        ));

        return (
            <AutoLabel input={input}>
                <Tooltip
                    borderRadius={8}
                    label={displayDirectory}
                    maxW="auto"
                    openDelay={500}
                    px={2}
                    py={0}
                >
                    <InputGroup
                        size="sm"
                        onContextMenu={menu.onContextMenu}
                        onDragOver={onDragOver}
                        onDrop={onDrop}
                    >
                        <InputLeftElement pointerEvents="none">
                            <Icon
                                as={BsFolderPlus}
                                m={0}
                            />
                        </InputLeftElement>
                        <Input
                            isReadOnly
                            borderRadius="lg"
                            cursor="pointer"
                            disabled={isLocked || isConnected}
                            draggable={false}
                            htmlSize={1}
                            placeholder="Click to select..."
                            size="sm"
                            textOverflow="ellipsis"
                            value={displayDirectory ?? ''}
                            // eslint-disable-next-line @typescript-eslint/no-misused-promises
                            onClick={onButtonClick}
                        />
                    </InputGroup>
                </Tooltip>
            </AutoLabel>
        );
    }
);
