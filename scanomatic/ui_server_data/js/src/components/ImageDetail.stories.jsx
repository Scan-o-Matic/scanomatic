import React from 'react';
import { storiesOf } from '@storybook/react';
import ImageDetail from './ImageDetail';

import imageUri from '../fixtures/fullres-scan.png';

storiesOf('ImageDetail', module)
    .add('View detail, cross hair', () => (
        <ImageDetail
            imageUri={imageUri}
            x={2696}
            y={2996}
            width={201}
            height={201}
            crossHair
        />
    ))
    .add('View detail, no cross hair', () => (
        <ImageDetail
            imageUri={imageUri}
            x={2696}
            y={2996}
            width={101}
            height={101}
        />
    ))
    .add('Partial out of bounds neg numbers, cross hair', () => (
        <ImageDetail
            imageUri={imageUri}
            x={224}
            y={316}
            width={501}
            height={701}
            crossHair
        />
    ))
    .add('Partial out of bounds overflow numbers, cross hair', () => (
        <ImageDetail
            imageUri={imageUri}
            x={4800}
            y={6000}
            width={101}
            height={101}
            crossHair
        />
    ));
