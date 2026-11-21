import { PackageSettings, SchemaId, WindowSize } from '../common-types';

/**
 * What a large image preview shows where there is no image: the theme's colour (with a checkerboard
 * behind transparent images), or a solid white or black that transparent pixels show too.
 */
export type ImagePreviewBackground = 'theme' | 'white' | 'black';

export interface ChainnerSettings {
    // renderer
    theme: string;
    language: string;
    checkForUpdatesOnStartup: boolean;
    startupTemplate: string;
    animateChain: boolean;
    snapToGrid: boolean;
    snapToGridAmount: number;
    viewportExportPadding: number;
    showMinimap: boolean;
    imagePreviewBackground: ImagePreviewBackground;

    experimentalFeatures: boolean;
    hardwareAcceleration: boolean;
    allowMultipleInstances: boolean;

    lastWindowSize: WindowSize;

    favoriteNodes: readonly SchemaId[];

    packageSettings: PackageSettings;

    storage: Readonly<Record<string, unknown>>;
}

export const defaultSettings: Readonly<ChainnerSettings> = {
    // renderer
    theme: 'default-dark',
    language: 'en',
    checkForUpdatesOnStartup: true,
    startupTemplate: '',
    animateChain: true,
    snapToGrid: false,
    snapToGridAmount: 16,
    viewportExportPadding: 20,
    showMinimap: false,
    imagePreviewBackground: 'theme',

    experimentalFeatures: false,
    hardwareAcceleration: false,
    allowMultipleInstances: false,

    lastWindowSize: {
        maximized: false,
        width: 1280,
        height: 720,
    },

    favoriteNodes: [],

    packageSettings: {},

    storage: {},
};
