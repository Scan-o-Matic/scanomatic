import { mount } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import { loadImage } from '../helpers';
import toLookLikeImage from '../helpers/toLookLikeImage';

import imageUri from '../fixtures/fullres-scan.png';
import expectedImageDetailCrosshair from '../fixtures/expectedImageDetailCrosshair.png';
import expectedImageDetailNoCrosshair from '../fixtures/expectedImageDetailNoCrosshair.png';

import ImageDetail from './ImageDetail';

const WAIT_FOR_IMAGE_LOAD = 10000;

describe('<ImageDetail />', () => {
    let wrapper;
    const onLoaded = jasmine.createSpy('onLoaded');
    const props = {
        x: 2696 + 75,
        y: 2996,
        width: 201,
        height: 201,
        crossHair: true,
        imageUri,
        onLoaded,
    };

    beforeEach(() => {
        onLoaded.calls.reset();
        jasmine.addMatchers({ toLookLikeImage });
    });

    describe('with crosshair', () => {
        beforeEach((done) => {
            wrapper = mount(<ImageDetail {...props} />);
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

        it('renders a canvas', () => {
            expect(wrapper.find('canvas').exists(0)).toBeTruthy();
        });

        it('has the expected dimensions', () => {
            const canvas = wrapper.find('canvas');
            expect(canvas.prop('width')).toEqual(props.width);
            expect(canvas.prop('height')).toEqual(props.height);
        });

        it('renders correctly', () => {
            const { canvas } = wrapper.instance();
            return loadImage(expectedImageDetailCrosshair).then((expected) => {
                expect(canvas).toLookLikeImage(expected);
            });
        }, WAIT_FOR_IMAGE_LOAD);
    });

    describe('without crosshair', () => {
        beforeEach((done) => {
            wrapper = mount(<ImageDetail {...props} crossHair={false} />);
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

        it('renders correctly', () => {
            const { canvas } = wrapper.instance();
            return loadImage(expectedImageDetailNoCrosshair).then((expected) => {
                expect(canvas).toLookLikeImage(expected);
            });
        }, WAIT_FOR_IMAGE_LOAD);
    });
});
