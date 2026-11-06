import { memo } from 'react';

interface TransElementProps {
    element: JSX.Element;
}

const renderElement = ({ element }: TransElementProps) => element;

/**
 * For a self-closing tag in a translation (`<icon/>`), react-i18next's `Trans`
 * replaces the component's children with the tag's empty content. Passing
 * `<TransElement element={...} />` as the component keeps the element as is.
 */
export const TransElement = memo(renderElement);
