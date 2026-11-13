import { useCallback, useMemo } from 'react';
import { SchemaId } from '../../common/common-types';
import { useMemoObject } from './useMemo';
import { useMutSetting } from './useSettings';

export interface UseNodeFavorites {
    readonly favorites: ReadonlySet<SchemaId>;
    readonly toggleFavorite: (schemaId: SchemaId) => void;
}

export const useNodeFavorites = () => {
    const [favoritesArray, setFavorites] = useMutSetting('favoriteNodes');

    const favorites = useMemo(() => new Set(favoritesArray), [favoritesArray]);

    const toggleFavorite = useCallback(
        (schemaId: SchemaId) => {
            setFavorites((prev) => {
                if (prev.includes(schemaId)) {
                    return prev.filter((id) => id !== schemaId);
                }
                return [...prev, schemaId];
            });
        },
        [setFavorites]
    );

    return useMemoObject<UseNodeFavorites>({
        favorites,
        toggleFavorite,
    });
};
