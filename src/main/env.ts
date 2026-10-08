import os from 'os';

export const isMac = process.platform === 'darwin';
const cpuModel = os.cpus()[0]?.model || null;
export const isArmMac: boolean = isMac && !!cpuModel && /Apple M\d/i.test(cpuModel);

const env = { ...process.env };
// These change where Python finds its standard library and modules. Another Python's settings
// break chaiNNer's own: e.g. a PYTHONPATH with Python 3.12's DLLs folder makes `import unicodedata`
// load a module built for python312.dll, PYTHONPLATLIBDIR hides the runtime's DLLs folder, and
// PYTHONSAFEPATH stops run.py from importing the modules next to it.
delete env.PYTHONHOME;
delete env.PYTHONPATH;
delete env.PYTHONPLATLIBDIR;
delete env.PYTHONSAFEPATH;
// Disable user site-packages to prevent chaiNNer from using global Python packages
// This ensures packages are installed in chaiNNer's isolated environment
env.PYTHONNOUSERSITE = '1';
export const sanitizedEnv = env;
