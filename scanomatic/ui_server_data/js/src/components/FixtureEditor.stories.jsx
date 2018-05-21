import React from 'react';
import { storiesOf } from '@storybook/react';
import { action } from '@storybook/addon-actions';
import FixtureEditor from './FixtureEditor';

import imageUri from '../fixtures/fullres-scan.png';
import '../../../style/bootstrap.css';
import '../../../style/fixtures-new.css';

const editActions = {
    onAreaStart: action('area-start'),
    onAreaEnd: action('area-end'),
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

storiesOf('FixtureEditor', module)
    .add('Grayscale detected', () => (
        <FixtureEditor
            imageUri={imageUri}
            {...markerAndAreas}
            editActions={editActions}
        />
    ));
