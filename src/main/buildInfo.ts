import { app } from 'electron/main';
import { BuildInfo } from '../common/version';

// BUILD_* are embedded by Vite (getBuildInfoDefine in vite/base.config.ts)
export const buildInfo: BuildInfo = {
    tree: BUILD_TREE,
    date: BUILD_DATE,
    dirty: BUILD_DIRTY,
    packaged: app.isPackaged,
};
