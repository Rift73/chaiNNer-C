import { DownloadIcon } from '@chakra-ui/icons';
import {
    AlertDialog,
    AlertDialogBody,
    AlertDialogContent,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogOverlay,
    Button,
    HStack,
    Heading,
    IconButton,
    Image,
    Tag,
    Tooltip,
    useDisclosure,
} from '@chakra-ui/react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useContext } from 'use-context-selector';
import { log } from '../../../common/log';
import { chainnerCVersion, isDevelopmentBuild } from '../../../common/version';
import logo from '../../../public/icons/png/256x256.png';
import { buildInfo } from '../../appConstants';
import { AlertBoxContext, AlertType } from '../../contexts/AlertBoxContext';
import {
    GitHubRelease,
    getLatestVersionIfUpdateAvailable,
    getReleaseTitle,
    getUpdateKey,
} from '../../helpers/github';
import { useIpcRendererListener } from '../../hooks/useIpcRendererListener';
import { useSettings } from '../../hooks/useSettings';
import { useStored } from '../../hooks/useStored';
import { ipcRenderer } from '../../safeIpc';

// One automatic check per start, also when the header is shown again after a backend restart.
let automaticCheckStarted = false;

// The update check only announces a release: nothing is downloaded or installed.
export const AppInfo = memo(() => {
    const { t } = useTranslation();
    const { checkForUpdatesOnStartup } = useSettings();
    const { sendAlert } = useContext(AlertBoxContext);

    const [updateVersion, setUpdateVersion] = useState<GitHubRelease>();

    // about 10 s after the window is ready; a failure is logged, not shown or retried
    useEffect(() => {
        if (automaticCheckStarted || !checkForUpdatesOnStartup || isDevelopmentBuild(buildInfo)) {
            return undefined;
        }
        const timeout = setTimeout(() => {
            automaticCheckStarted = true;
            getLatestVersionIfUpdateAvailable(chainnerCVersion, buildInfo)
                .then((release) => {
                    if (release) {
                        setUpdateVersion(release);
                    }
                })
                .catch((error) => {
                    log.warn(`Could not check for updates: ${String(error)}`);
                });
        }, 10_000);
        return () => clearTimeout(timeout);
    }, [checkForUpdatesOnStartup]);

    const { isOpen: isAlertOpen, onOpen: onAlertOpen, onClose: onAlertClose } = useDisclosure();

    const [lastIgnoredUpdate, setLastIgnoredUpdate] = useStored<string | undefined>(
        'ignored-update',
        undefined
    );

    const leastDestructiveRef = useRef(null);

    useEffect(() => {
        if (updateVersion && lastIgnoredUpdate !== getUpdateKey(updateVersion)) {
            onAlertOpen();
        }
    }, [lastIgnoredUpdate, onAlertOpen, updateVersion]);

    // Help > Check for Updates...: also for a release the user chose not to be shown again
    useIpcRendererListener(
        'check-for-updates',
        useCallback(() => {
            getLatestVersionIfUpdateAvailable(chainnerCVersion, buildInfo)
                .then((release) => {
                    if (release) {
                        setUpdateVersion(release);
                        onAlertOpen();
                    } else {
                        sendAlert({
                            type: AlertType.INFO,
                            title: t('header.checkForUpdates', 'Check for Updates'),
                            message: `${t(
                                'header.upToDate',
                                "You're up to date: chaiNNer-C v{{version}}.",
                                { version: chainnerCVersion }
                            )}${
                                isDevelopmentBuild(buildInfo)
                                    ? ` ${t(
                                          'header.developmentBuildUpdates',
                                          'This is a development build: only a newer version counts as an update.'
                                      )}`
                                    : ''
                            }`,
                        });
                    }
                })
                .catch((error) => {
                    log.warn(`Could not check for updates: ${String(error)}`);
                    sendAlert({
                        type: AlertType.WARN,
                        title: t('header.checkForUpdates', 'Check for Updates'),
                        message: t(
                            'header.couldNotCheckForUpdates',
                            'Could not check for updates: {{error}}',
                            { error: String(error) }
                        ),
                    });
                });
        }, [onAlertOpen, sendAlert, t])
    );

    const updateTitle = updateVersion && getReleaseTitle(updateVersion, chainnerCVersion);

    return (
        <>
            <HStack>
                <Image
                    boxSize="36px"
                    draggable={false}
                    src={logo}
                />
                <Heading
                    display={{ base: 'none', lg: 'inherit' }}
                    size="md"
                >
                    chaiNNer
                </Heading>
                <Tag>v{chainnerCVersion}</Tag>
                {updateVersion && (
                    <Tooltip
                        closeOnClick
                        closeOnMouseDown
                        borderRadius={8}
                        label={t('header.updateAvailable', 'Update available ({{version}})', {
                            version: updateTitle ?? '',
                        })}
                        px={2}
                        py={1}
                    >
                        <IconButton
                            aria-label={t('header.updateAvailableAria', 'Update available')}
                            colorScheme="green"
                            icon={<DownloadIcon />}
                            variant="ghost"
                            onClick={onAlertOpen}
                        />
                    </Tooltip>
                )}
            </HStack>
            <AlertDialog
                isCentered
                isOpen={isAlertOpen}
                leastDestructiveRef={leastDestructiveRef}
                onClose={onAlertClose}
            >
                <AlertDialogOverlay>
                    <AlertDialogContent bgColor="var(--chain-editor-bg)">
                        <AlertDialogHeader
                            fontSize="lg"
                            fontWeight="bold"
                        >
                            {t('header.updateAvailableTitle', 'Update Available ({{version}})', {
                                version: updateTitle,
                            })}
                        </AlertDialogHeader>

                        <AlertDialogBody>
                            {t(
                                'header.updateAvailableOpenRelease',
                                'There is an update available: {{version}}. Would you like to open its release page?',
                                { version: updateTitle }
                            )}
                        </AlertDialogBody>

                        <AlertDialogFooter>
                            <HStack>
                                <Button
                                    ref={leastDestructiveRef}
                                    variant="ghost"
                                    onClick={onAlertClose}
                                >
                                    {t('common.close', 'Close')}
                                </Button>
                                <Button
                                    onClick={() => {
                                        if (updateVersion) {
                                            setLastIgnoredUpdate(getUpdateKey(updateVersion));
                                        }
                                        onAlertClose();
                                    }}
                                >
                                    {t('header.ignoreRelease', "Don't show again for this release")}
                                </Button>
                                <Button
                                    colorScheme="green"
                                    onClick={() => {
                                        if (updateVersion) {
                                            ipcRenderer
                                                .invoke('open-url', updateVersion.url)
                                                .catch(() => {
                                                    log.error('Failed to open the release page');
                                                });
                                        }
                                        onAlertClose();
                                    }}
                                >
                                    {t('header.openReleasePage', 'Open release page')}
                                </Button>
                            </HStack>
                        </AlertDialogFooter>
                    </AlertDialogContent>
                </AlertDialogOverlay>
            </AlertDialog>
        </>
    );
});
