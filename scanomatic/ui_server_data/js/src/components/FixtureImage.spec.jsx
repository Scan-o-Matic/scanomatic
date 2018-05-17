import { mount } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import { loadImage } from '../../src/helpers';

import FixtureImage from './FixtureImage';

import imageUri from '../fixtures/fullres-scan.png';

const WAIT_FOR_IMAGE_LOAD = 5000;

describe('<FixtureImage />', () => {
    const onAreaStart = jasmine.createSpy('onAreaStart');
    const onAreaEnd = jasmine.createSpy('onAreaEnd');
    const onMouse = jasmine.createSpy('onMouse');
    const onClick = jasmine.createSpy('onClick');
    const onLoaded = jasmine.createSpy('onLoaded');

    let wrapper;

    beforeEach(() => {
        wrapper = mount((
            <FixtureImage
                imageUri={imageUri}
                markers={[
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
                ]}
                areas={[
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
                ]}
                onAreaStart={onAreaStart}
                onAreaEnd={onAreaEnd}
                onMouse={onMouse}
                onClick={onClick}
                onLoaded={onLoaded}
            />
        ));
        onAreaStart.calls.reset();
        onAreaEnd.calls.reset();
        onMouse.calls.reset();
        onClick.calls.reset();
        onLoaded.calls.reset();
    });

    it('renders two canvas', () => {
        expect(wrapper.find('canvas').exists()).toBeTruthy();
        expect(wrapper.find('canvas').length).toEqual(2);
    });

    it('triggers onLoaded when image is loaded', () => {
        waitsFor(() => onLoaded.calls.any(), WAIT_FOR_IMAGE_LOAD);
    });

    it('image canvas should be 480x600px', () => {

    });

    it('renders the image with flipped x-axis', (done) => {
        waitsFor(() => onLoaded.calls.any(), WAIT_FOR_IMAGE_LOAD);
        runs(() => {
            const canvas = wrapper.instance().imageCanvas;
            loadImage(imageUri).then((expected) => {
                expect(canvas).toLookLikeImage(expected);
                done();
            });
        });
    });

    it('renders the overlays', () => {

    });

    it('rerenders the overlays on new props', () => {

    });

    it('renders error on bad image', () => {

    });

    it('renders error on new image with bad uri', () => {

    });

    it('updates canvases on new image with new size', () => {

    });

    it('converts to image coordinates', () => {

    });
});
