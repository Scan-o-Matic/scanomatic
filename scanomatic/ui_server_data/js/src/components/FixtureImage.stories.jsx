import React from 'react';
import { storiesOf } from '@storybook/react';
import { action } from '@storybook/addon-actions';
import FixtureImage from './FixtureImage';

import imageUri from '../fixtures/fullres-scan.png';
import '../../../style/bootstrap.css';

const actions = {
    onAreaStart: action('area-start'),
    onAreaEnd: action('area-end'),
    onMouse: action('mouse'),
    onClick: action('click'),
};

const markerAndAreas = {
    markers: [
        {
            x: 2696,
            y: 2996,
        },
        {
            x: 224,
            y: 316,
        },
        {
            x: 388,
            y: 5744,
        },
    ],
    areas: [
        {
            name: '1',
            rect: {
                x: 2860,
                y: 36,
                w: 4780 - 2860,
                h: 2824 - 36,
            },
        },
        {
            name: 'G',
            rect: {
                x: 264,
                y: 2532,
                w: 400 - 264,
                h: 3296 - 2532,
            },
        },
    ],
};

storiesOf('FixtureImage', module)
    .add('Edit mode', () => (
        <FixtureImage
            imageUri={imageUri}
            {...markerAndAreas}
            {...actions}
        />
    ))
    .add('View mode', () => (
        <FixtureImage
            imageUri={imageUri}
            {...markerAndAreas}
        />
    ))
    .add('Missing image', () => (
        <FixtureImage
            imageUri="http://localhost/badimage/please"
            {...markerAndAreas}
        />
    ));
