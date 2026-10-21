import { createHash } from 'crypto';
import fs from 'fs/promises';
import { basename, dirname, join, resolve } from 'path';
import { EdgeData, FileOpenResult, NodeData, Version } from '../common/common-types';
import { log } from '../common/log';
import { currentMigration, migrate } from '../common/migrations';
import { versionGt } from '../common/version';
import { readTextFile, writeFileAtomic } from './util';
import type { Edge, Node, Viewport } from 'reactflow';

export interface SaveData {
    nodes: Node<NodeData>[];
    edges: Edge<EdgeData>[];
    viewport: Viewport;
}

export interface ParsedSaveData extends SaveData {
    tamperedWith: boolean;
}

export interface RawSaveFile {
    version: Version;
    content: SaveData;
    timestamp?: string;
    checksum?: string;
    migration?: number;
}

const hash = (value: string): string => {
    return createHash('md5').update(value).digest('hex');
};

export class SaveFile {
    private static pendingReservation: Promise<void> = Promise.resolve();

    private static readonly pendingWrites = new Map<string, Promise<void>>();

    static parse(value: string): ParsedSaveData {
        if (!/^\s*\{/.test(value)) {
            // base64 decode
            // eslint-disable-next-line no-param-reassign
            value = Buffer.from(value, 'base64').toString('utf-8');
        }

        const rawData = JSON.parse(value) as RawSaveFile | (SaveData & { version?: never });

        let data: SaveData;
        let tamperedWith = false;
        if (rawData.version) {
            const { version, content, checksum, migration } = rawData;
            if (checksum !== undefined) {
                // checksum is present
                tamperedWith = checksum !== hash(JSON.stringify(content));
            } else {
                // checksum was added after v0.6.1, so any saves >0.6.1 without a checksum have
                // been tampered with
                tamperedWith = versionGt(version, '0.6.1') || (migration ?? 0) > 4;
            }

            data = migrate(version, content, migration) as SaveData;
        } else {
            // Legacy files
            data = migrate(null, rawData) as SaveData;
        }

        return {
            ...data,
            tamperedWith,
        };
    }

    static async read(path: string): Promise<ParsedSaveData> {
        return SaveFile.parse(await readTextFile(path));
    }

    static stringify(content: SaveData, version: Version): string {
        const { nodes, edges, viewport } = content;

        const sanitizedContent: RawSaveFile['content'] = {
            nodes: nodes.map(
                (n): Node<NodeData> => ({
                    data: {
                        schemaId: n.data.schemaId,
                        inputData: n.data.inputData,
                        inputHeight: n.data.inputHeight,
                        outputHeight: n.data.outputHeight,
                        nodeWidth: n.data.nodeWidth,
                        id: n.data.id,
                        isDisabled: n.data.isDisabled,
                        isLocked: n.data.isLocked,
                        isCollapsed: n.data.isCollapsed,
                        isPassthrough: n.data.isPassthrough,
                        nodeName: n.data.nodeName,
                    },
                    id: n.id,
                    position: n.position,
                    type: n.type,
                    selected: n.selected,
                    height: n.height,
                    width: n.width,
                    zIndex: n.zIndex,
                })
            ),
            edges: edges.map((e): Edge<EdgeData> => ({ ...e, data: {} })),
            viewport,
        };

        const data: Required<RawSaveFile> = {
            version,
            content: sanitizedContent,
            timestamp: new Date().toISOString(),
            checksum: hash(JSON.stringify(sanitizedContent)),
            migration: currentMigration,
        };
        return JSON.stringify(data);
    }

    /**
     * Writes the chain through a temporary file and a rename, so a crash or a full disk leaves the
     * old chain or the new one, never a partial file. A link keeps pointing to the chain: the file
     * it points to gets the new content. A file that refuses to be replaced (another program holds
     * it open without sharing delete access, e.g. a sync client) is written in place instead.
     */
    static async write(path: string, saveData: SaveData, version: Version): Promise<void> {
        const content = SaveFile.stringify(saveData, version);
        const absolutePath = resolve(path);
        const previousReservation = SaveFile.pendingReservation;
        let finishReservation = () => {};
        // Reserve arrival order before realpath, including requests through different aliases.
        SaveFile.pendingReservation = new Promise<void>((resolveReservation) => {
            finishReservation = resolveReservation;
        });
        let writing: Promise<void>;
        try {
            await previousReservation;
            const target = await fs.realpath(absolutePath).catch(async () => {
                const parent = dirname(absolutePath);
                const directory = await fs.realpath(parent).catch(() => parent);
                return join(directory, basename(absolutePath));
            });
            const key = process.platform === 'win32' ? target.toLowerCase() : target;
            const previous = SaveFile.pendingWrites.get(key);
            let finish = () => {};
            const done = new Promise<void>((resolveWrite) => {
                finish = resolveWrite;
            });
            SaveFile.pendingWrites.set(key, done);
            writing = (async () => {
                try {
                    await previous;
                    try {
                        await writeFileAtomic(target, content);
                    } catch (error) {
                        const { code } = error as NodeJS.ErrnoException;
                        if (code !== 'EPERM' && code !== 'EBUSY') throw error;
                        log.warn(`Unable to replace ${target} (${code}), writing it in place.`);
                        await fs.writeFile(target, content, 'utf-8');
                    }
                } finally {
                    // Failure rejects its caller while releasing the next save of this target.
                    finish();
                    if (SaveFile.pendingWrites.get(key) === done) {
                        SaveFile.pendingWrites.delete(key);
                    }
                }
            })();
        } finally {
            // Only resolution/reservation is global; writes to unrelated targets can overlap.
            finishReservation();
        }
        await writing;
    }
}

export const openSaveFile = async (path: string): Promise<FileOpenResult<ParsedSaveData>> => {
    try {
        const saveData = await SaveFile.read(path);
        return { kind: 'Success', path, saveData };
    } catch (error) {
        log.error(error);
        return { kind: 'Error', path, error: String(error) };
    }
};
