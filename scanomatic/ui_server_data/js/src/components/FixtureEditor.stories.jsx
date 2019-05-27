import React from 'react';
import PropTypes from 'prop-types';
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

const editorActions = {
    onFinalize: action('finalize'),
    onResetAreas: action('reset-areas'),
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

class LiveUpdater extends React.Component {
    constructor(props) {
        super(props);
        this.state = { idx: 0 };
    }

    componentDidMount() {
        const { frequency } = this.props;
        setInterval(() => this.setState({ idx: this.state.idx + 1 }), frequency);
    }

    render() {
        const {
            Target, updateFunction, targetProps,
        } = this.props;
        const { idx } = this.state;
        return <Target {...updateFunction(targetProps, idx)} />;
    }
}

LiveUpdater.propTypes = {
    Target: PropTypes.element.isRequired,
    updateFunction: PropTypes.func.isRequired,
    frequency: PropTypes.number.isRequired,
    targetProps: PropTypes.shape({}),
};

LiveUpdater.defaultProps = {
    targetProps: {},
};

storiesOf('FixtureEditor', module)
    .add('Grayscale not detected', () => (
        <FixtureEditor
            imageUri={imageUri}
            {...markerAndAreas}
            editActions={editActions}
            scannerName="Monsterous Magpie"
            {...editorActions}
        />
    ))
    .add('Grayscale not detected, no plates', () => (
        <FixtureEditor
            imageUri={imageUri}
            markers={markerAndAreas.markers}
            areas={[]}
            editActions={editActions}
            scannerName="Monsterous Magpie"
            {...editorActions}
        />
    ))
    .add('Grayscale detected', () => (
        <FixtureEditor
            imageUri={imageUri}
            {...markerAndAreas}
            editActions={editActions}
            scannerName="Monsterous Magpie"
            grayscaleDetection={{
                referenceValues: [0, 10, 30, 60, 95],
                pixelValues: [10, 30, 100, 150, 240],
            }}
            {...editorActions}
            validFixture
        />
    ))
    .add('Live redetect Grayscale detected', () => (
        <LiveUpdater
            Target={FixtureEditor}
            updateFunction={(props, idx) => Object.assign(
                {},
                props,
                {
                    grayscaleDetection: {
                        referenceValues: [0, 10, 30, 60, 95],
                        pixelValues: [10, 20, 80, 200, 4 * idx],
                    },
                },
            )}
            targetProps={{
                imageUri,
                editActions,
                scannerName: 'Monsterous Magpie',
                grayscaleDetection: {
                    referenceValues: [0, 10, 30, 60, 95],
                    pixelValues: [10, 30, 100, 150, 240],
                },
                ...markerAndAreas,
                ...editorActions,
            }}
            frequency={500}
        />
    ));
