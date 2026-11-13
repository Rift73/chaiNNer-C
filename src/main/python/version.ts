import { PythonInfo } from '../../common/common-types';
import { parse, versionGte } from '../../common/version';
import { promisifiedSpawn } from '../childProc';

/** The Python executable and its version; rejects when it does not run or is older than 3.7. */
export const checkPython = async (python: string): Promise<PythonInfo> => {
    const version = parse(await promisifiedSpawn(python, ['--version']));
    if (!versionGte(version, '3.7.0')) {
        throw new Error(`${python} is Python ${version}, and chaiNNer needs Python 3.7 or newer.`);
    }
    return { python, version };
};
