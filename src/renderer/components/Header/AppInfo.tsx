import { HStack, Heading, Image, Tag } from '@chakra-ui/react';
import { memo } from 'react';
import { chainnerCVersion } from '../../../common/version';
import logo from '../../../public/icons/png/256x256.png';

// chaiNNer-C has no update check: the logo, the title and chaiNNer-C's version.
export const AppInfo = memo(() => (
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
    </HStack>
));
