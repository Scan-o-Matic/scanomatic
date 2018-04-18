import React from 'react';
import { mount } from 'enzyme';

import './enzyme-setup';
import Plate from '../../src/components/Plate';
import { loadImage } from '../../src/helpers';
import expectedPlateImage from '../fixtures/expectedPlate.png';
import expectedGriddedPlateImage from '../fixtures/expectedGriddedPlate.png';
import expectedGriddedPlateSelectFirstImage from '../fixtures/expectedGriddedPlateSelectFirst.png';
import expectedGriddedPlateSelectSecondImage from '../fixtures/expectedGriddedPlateSelectSecond.png';
import testPlateImage from '../fixtures/testPlate.png';

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
