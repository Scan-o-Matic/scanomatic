import { mount, shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import { loadImage } from '../../src/helpers';

import FixtureImage from './FixtureImage';

import imageUri from '../fixtures/fullres-scan.png';

const WAIT_FOR_IMAGE_LOAD = 10000;

const toLookLikeImage = (util, customEqualityTesters) => ({
    compare: (actual, expected) => {
        const expectedCanvas = document.createElement('canvas');
        expectedCanvas.width = expected.naturalWidth;
        expectedCanvas.height = expected.naturalHeight;
        expectedCanvas.getContext('2d').drawImage(expected, 0, 0);
        const actualImageData = actual.getContext('2d')
            .getImageData(0, 0, actual.width, actual.height);
        const expectedImageData = expectedCanvas.getContext('2d')
            .getImageData(0, 0, actual.width, actual.height);
        const result = {};
        result.pass = util.equals(actualImageData, expectedImageData, customEqualityTesters);

        const actualDataURL = actual.toDataURL();
        const expectedDataURL = expectedCanvas.toDataURL();
        if (result.pass) {
            result.message = "Expected canvas to look different";
        } else {
            result.message = "Expected canvas to look the same";
        }
        result.message += "\n\texpected: " + expectedDataURL;
        result.message += "\n\tactual: " + actualDataURL;
        return result;
    },
});

describe('<FixtureImage />', () => {
    const onAreaStart = jasmine.createSpy('onAreaStart');
    const onAreaEnd = jasmine.createSpy('onAreaEnd');
    const onMouse = jasmine.createSpy('onMouse');
    const onClick = jasmine.createSpy('onClick');
    const onLoaded = jasmine.createSpy('onLoaded');

    let wrapper;
    const markers = [
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
    ];
    const areas = [
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
    ];

    beforeEach(() => {
        onAreaStart.calls.reset();
        onAreaEnd.calls.reset();
        onMouse.calls.reset();
        onClick.calls.reset();
        onLoaded.calls.reset();
    });

    describe('before image loaded', () => {
        beforeEach(() => {
            wrapper = shallow((
                <FixtureImage
                    imageUri={imageUri}
                    markers={markers}
                    areas={areas}
                    onAreaStart={onAreaStart}
                    onAreaEnd={onAreaEnd}
                    onMouse={onMouse}
                    onClick={onClick}
                    onLoaded={onLoaded}
                />
            ));
        });

        it('has an alert indicating it is loading', () => {
            const alert = wrapper.find('.alert');
            expect(alert.exists()).toBeTruthy();
            expect(alert.text()).toContain('Loading...');
        });
    });

    describe('after image loaded', () => {
        beforeEach((done) => {
            jasmine.addMatchers({ toLookLikeImage });
            wrapper = mount((
                <FixtureImage
                    imageUri={imageUri}
                    markers={markers}
                    areas={areas}
                    onAreaStart={onAreaStart}
                    onAreaEnd={onAreaEnd}
                    onMouse={onMouse}
                    onClick={onClick}
                    onLoaded={onLoaded}
                />
            ));
            const loaded = () => {
                if (onLoaded.calls.any()) {
                    wrapper.update();
                    done();
                } else {
                    setTimeout(loaded, 100);
                }
            };
            loaded();
        });

        it('has no alert', () => {
            expect(wrapper.find('.alert').exists()).toBeFalsy();
        });

        it('renders two canvas', () => {
            expect(wrapper.find('canvas').length).toEqual(2);
        });

        it('renders canvases with dimensions 480x600px', () => {
            const canvases = wrapper.find('canvas');
            expect(canvases.at(0).prop('width')).toEqual(480);
            expect(canvases.at(1).prop('width')).toEqual(480);
            expect(canvases.at(0).prop('height')).toEqual(600);
            expect(canvases.at(1).prop('height')).toEqual(600);
        });

        it('renders canvases ontop of eachother', () => {
            const canvases = wrapper.find('canvas');
            expect(canvases.at(0).prop('style')).toEqual({
                position: 'absolute',
                zIndex: 1,
                left: 0,
                top: 0,
            });
            expect(canvases.at(1).prop('style')).toEqual({
                position: 'absolute',
                zIndex: 2,
                left: 0,
                top: 0,
            });
        });

        it('renders the image with flipped x-axis', () => {
            const canvas = wrapper.instance().imageCanvas;
            return loadImage(imageUri).then((expected) => {
                console.log('loaded', imageUri);
                expect(canvas).toLookLikeImage(expected);
            });
        }, WAIT_FOR_IMAGE_LOAD);

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
});
