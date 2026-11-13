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
// The last update's result is reported once per start.
let updateResultTaken = false;

// The update check announces a release; Update now downloads, checks and installs it, and
// chaiNNer-C starts again.
export const AppInfo = memo(() => {
    const { t } = useTranslation();
    const { checkForUpdatesOnStartup } = useSettings();
    const { sendAlert } = useContext(AlertBoxContext);

    const [updateVersion, setUpdateVersion] = useState<GitHubRelease>();
    // the download's percentage while Update now runs
    const [updateProgress, setUpdateProgress] = useState<number>();

    useEffect(() => {
        if (updateResultTaken) {
            return;
        }
        updateResultTaken = true;
        ipcRenderer
            .invoke('take-update-result')
            .then((result) => {
                if (!result) {
                    return;
                }
                sendAlert(
                    result.ok
                        ? {
                              type: AlertType.INFO,
                              title: t('header.updateInstalledTitle', 'Updated'),
                              message: t(
                                  'header.updateInstalled',
                                  'chaiNNer-C was updated to {{release}}.',
                                  { release: result.release }
                              ),
                          }
                        : {
                              type: AlertType.WARN,
                              title: t('header.updateFailedTitle', 'Update not installed'),
                              message: t(
                                  'header.updateInstallFailed',
                                  'The update to {{release}} could not be installed: {{message}} Its zip is kept at {{zip}}.',
                                  {
                                      release: result.release,
                                      message: result.message,
                                      zip: result.zip,
                                  }
                              ),
                          }
                );
            })
            .catch((error) => {
                log.warn(`Could not read the last update's result: ${String(error)}`);
            });
    }, [sendAlert, t]);

    useIpcRendererListener(
        'update-download-progress',
        useCallback((_, percentage: number) => setUpdateProgress(percentage), [])
    );

    const updateNow = useCallback(() => {
        setUpdateProgress(0);
        ipcRenderer
            .invoke('download-update')
            .then(() => ipcRenderer.invoke('install-update'))
            .then((installing) => {
                if (!installing) {
                    setUpdateProgress(undefined);
                }
            })
            .catch((error) => {
                setUpdateProgress(undefined);
                log.error(`Could not update: ${String(error)}`);
                sendAlert({
                    type: AlertType.ERROR,
                    title: t('header.updateFailedTitle', 'Update not installed'),
                    message: t('header.updateFailed', 'Could not update: {{error}}', {
                        error: String(error),
                    }),
                });
            });
    }, [sendAlert, t]);
    const updating = updateProgress !== undefined;

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
                            {updating
                                ? t(
                                      'header.updateDownloading',
                                      'Downloading {{version}}: {{percentage}} %. chaiNNer-C then closes, installs it and starts again.',
                                      {
                                          version: updateTitle,
                                          percentage: Math.floor(updateProgress),
                                      }
                                  )
                                : t(
                                      'header.updateAvailableInstall',
                                      'There is an update available: {{version}}. Update now downloads and installs it, and chaiNNer-C starts again; your settings and installed packages stay.',
                                      { version: updateTitle }
                                  )}
                        </AlertDialogBody>

                        <AlertDialogFooter>
                            <HStack>
                                <Button
                                    isDisabled={updating}
                                    ref={leastDestructiveRef}
                                    variant="ghost"
                                    onClick={onAlertClose}
                                >
                                    {t('common.close', 'Close')}
                                </Button>
                                <Button
                                    isDisabled={updating}
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
                                    isDisabled={updating}
                                    onClick={() => {
                                        if (updateVersion) {
                                            ipcRenderer
                                                .invoke('open-url', updateVersion.url)
                                                .catch((error) => {
                                                    log.error(
                                                        'Failed to open the release page',
                                                        error
                                                    );
                                                });
                                        }
                                        onAlertClose();
                                    }}
                                >
                                    {t('header.openReleasePage', 'Open release page')}
                                </Button>
                                <Button
                                    colorScheme="green"
                                    isLoading={updating}
                                    onClick={updateNow}
                                >
                                    {t('header.updateNow', 'Update now')}
                                </Button>
                            </HStack>
                        </AlertDialogFooter>
                    </AlertDialogContent>
                </AlertDialogOverlay>
            </AlertDialog>
        </>
    );
});
