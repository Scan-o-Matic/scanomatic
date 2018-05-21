import { mount, shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import { loadImage } from '../helpers';
import toLookLikeImage from '../helpers/toLookLikeImage';

import FixtureImage from './FixtureImage';

import imageUri from '../fixtures/fixtureImageForTest.png';
import expectedFixtureImageUri from '../fixtures/expectedFixtureImage.png';
import expectedFixtureOverlaysUri from '../fixtures/expectedFixtureOverlays.png';
import expectedFixtureOverlaysUpdatedUri from '../fixtures/expectedFixtureOverlays-updated.png';

const WAIT_FOR_IMAGE_LOAD = 10000;


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
            return loadImage(expectedFixtureImageUri).then((expected) => {
                expect(canvas).toLookLikeImage(expected);
            });
        }, WAIT_FOR_IMAGE_LOAD);

        it('renders the overlays', () => {
            const canvas = wrapper.instance().overlayCanvas;
            return loadImage(expectedFixtureOverlaysUri).then((expected) => {
                expect(canvas).toLookLikeImage(expected);
            });
        }, WAIT_FOR_IMAGE_LOAD);

        it('rerenders the overlays on new props', () => {
            const canvas = wrapper.instance().overlayCanvas;
            wrapper.setProps({
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
                    {
                        name: '2',
                        rect: {
                            x: 520,
                            y: 36,
                            w: 4780 - 2860,
                            h: 2824 - 36,
                        },
                    },
                ],
            });
            wrapper.update();
            return loadImage(expectedFixtureOverlaysUpdatedUri).then((expected) => {
                expect(canvas).toLookLikeImage(expected);
            });
        }, WAIT_FOR_IMAGE_LOAD);

        it('has an alert indicating it is loading if image URI updates', () => {
            wrapper.setProps({ imageUri: expectedFixtureImageUri });
            wrapper.update();
            const alert = wrapper.find('.alert');
            expect(alert.exists()).toBeTruthy();
            expect(alert.text()).toContain('Loading...');
        });

        it('converts to image coordinates', () => {

        });
    });

    describe('Loading bad image', () => {
        beforeEach((done) => {
            wrapper = mount((
                <FixtureImage
                    imageUri="/not/an/image"
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

        it('renders error', () => {
            expect(wrapper.find('.alert').exists()).toBeTruthy();
            expect(wrapper.find('.alert').text()).toEqual('Could not load fixture image!');
        }, WAIT_FOR_IMAGE_LOAD);
    });

    describe('Loading and then reloading bad image', () => {
        beforeEach((done) => {
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
            let expectedCalls = 1;
            const loaded = () => {
                if (onLoaded.calls.count() === expectedCalls) {
                    if (expectedCalls === 1) {
                        wrapper.setProps({ imageUri: '/not/an/image' });
                        wrapper.update();
                        expectedCalls = 2;
                        setTimeout(loaded, 100);
                    } else {
                        wrapper.update();
                        done();
                    }
                } else {
                    setTimeout(loaded, 100);
                }
            };
            loaded();
        });

        it('renders error', () => {
            expect(wrapper.find('.alert').exists()).toBeTruthy();
            expect(wrapper.find('.alert').text()).toEqual('Could not load fixture image!');
        });
    });

    describe('Loading and then loading other image', () => {
        beforeEach((done) => {
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
            let expectedCalls = 1;
            const loaded = () => {
                if (onLoaded.calls.count() === expectedCalls) {
                    if (expectedCalls === 1) {
                        wrapper.setProps({ imageUri: expectedFixtureImageUri });
                        wrapper.update();
                        expectedCalls = 2;
                        setTimeout(loaded, 100);
                    } else {
                        wrapper.update();
                        done();
                    }
                } else {
                    setTimeout(loaded, 100);
                }
            };
            loaded();
        });

        it('updates canvases on new image with new size', () => {
            const canvases = wrapper.find('canvas');
            expect(canvases.at(0).prop('width')).toEqual(48);
            expect(canvases.at(1).prop('width')).toEqual(48);
            expect(canvases.at(0).prop('height')).toEqual(60);
            expect(canvases.at(1).prop('height')).toEqual(60);
        }, WAIT_FOR_IMAGE_LOAD);
    });
});
