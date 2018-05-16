import React from 'react';
import { storiesOf } from '@storybook/react';
import { action } from '@storybook/addon-actions';
import FixtureImage from './FixtureImage';

import imageData from '../fixtures/fullres-scan.png';
const img = new Image();
img.src = imageData;

const actions = {
    onAreaStart: action('area-start'),
    onAreaEnd: action('area-end'),
    onMouse: action('mouse'),
};

storiesOf('FixtureImage', module)
    .add('Edit mode', () => (
        <FixtureImage
            image={img}
            markers={[]}
            areas={[]}
            {...actions}
        />
    ));
