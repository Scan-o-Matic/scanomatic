import React from 'react';
import { mount } from 'enzyme';

import './enzyme-setup';
import Plate from './Plate';
import { loadImage } from '../helpers';
import toLookLikeImage from '../helpers/toLookLikeImage';
import expectedPlateImage from '../fixtures/expectedPlate.png';
import expectedGriddedPlateImage from '../fixtures/expectedGriddedPlate.png';
import expectedGriddedPlateSelectFirstImage from '../fixtures/expectedGriddedPlateSelectFirst.png';
import expectedGriddedPlateSelectSecondImage from '../fixtures/expectedGriddedPlateSelectSecond.png';
import testPlateImage from '../fixtures/testPlate.png';

describe('<Plate />', () => {
    let image;
    const grid = [
        [[150, 150], [50, 50]],
        [[50, 150], [50, 150]],
    ];

    beforeEach((done) => {
        jasmine.addMatchers({ toLookLikeImage });
        loadImage(testPlateImage).then((img) => { image = img; done() });
    });

    it('should render a <canvas />', () => {
        const wrapper = mount(<Plate image={image} />);
        expect(wrapper.find('canvas').exists()).toBe(true);
    });

    it('should size <canvas /> based on the image', () => {
        const wrapper = mount(<Plate image={image} />);
        const canvas = wrapper.find('canvas');
        expect(canvas.prop('width')).toEqual(image.width * .2);
        expect(canvas.prop('height')).toEqual(image.height * .2);
    });

    it('should render the image in the <canvas />', (done) => {
        const wrapper = mount(<Plate image={image} />);
        const canvasElement = wrapper.instance().canvas;
        loadImage(expectedPlateImage).then((expected) => {
            expect(canvasElement).toLookLikeImage(expected);
            done();
        });
    });

    it('should render the grid on top of the image', (done) => {
        const wrapper = mount(<Plate image={image} grid={grid} />);
        const canvasElement = wrapper.instance().canvas;
        loadImage(expectedGriddedPlateImage).then((expected) => {
            expect(canvasElement).toLookLikeImage(expected);
            done();
        });
    });

    it('should render a circle around the selected colony', (done) => {
        const wrapper = mount(<Plate image={image} grid={grid} selectedColony={{ row: 0, col: 0 }} />);
        const canvasElement = wrapper.instance().canvas;
        loadImage(expectedGriddedPlateSelectFirstImage).then((expected) => {
            expect(canvasElement).toLookLikeImage(expected);
            done();
        });
    });

    it('should update the canvas when props change', (done) => {
        const wrapper = mount(<Plate image={image} grid={grid} selectedColony={{ row: 0, col: 0 }} />);
        wrapper.setProps({ selectedColony: { row: 0, col: 1 } });
        const canvasElement = wrapper.instance().canvas;
        loadImage(expectedGriddedPlateSelectSecondImage).then((expected) => {
            expect(canvasElement).toLookLikeImage(expected);
            done();
        });
    });
});
