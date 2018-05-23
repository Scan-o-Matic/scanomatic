import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';

import FixtureEditor from './FixtureEditor';

describe('<FixtureEditor />', () => {
    const onFinalize = jasmine.createSpy('onFinalize');
    const onResetAreas = jasmine.createSpy('onResetAreas');
    const props = {
        onFinalize,
        onResetAreas,
        scannerName: 'Invisible Ignaonodon',
        imageUri: 'breakthru.tiff',
        markers: [{ x: 5, y: 100 }],
        areas: [{
            name: 'G',
            rect: {
                x: 1000, y: 2402, w: 11, h: 231,
            },
        }],
        editActions: {
            onAreaStart: () => {},
            onAreaEnd: () => {},
            onClick: () => {},
        },
    };
    let wrapper;

    describe('with grayscale detection', () => {
        const grayscaleDetection = {
            referenceValues: [1, 2, 3],
            pixelValues: [255, 132, 10],
        };

        beforeEach(() => {
            wrapper = shallow(<FixtureEditor {...props} grayscaleDetection={grayscaleDetection} />);
        });

        it('should set the scanner name as heading', () => {
            expect(wrapper.find('h2').exists()).toBeTruthy();
            expect(wrapper.find('h2').text()).toEqual(props.scannerName);
        });

        it('sets the grayscale type', () => {
            expect(wrapper.find('.grayscale-type').exists()).toBeTruthy();
            expect(wrapper.find('.grayscale-type').prop('value'))
                .toEqual('silverfast');
        });

        describe('<FixtureImage />', () => {
            let fixtureImage;

            beforeEach(() => {
                fixtureImage = wrapper.find('FixtureImage');
            });

            it('renders', () => {
                expect(fixtureImage.exists()).toBeTruthy();
            });

            it('passes the imageUri', () => {
                expect(fixtureImage.prop('imageUri')).toEqual(props.imageUri);
            });

            it('passes the markers', () => {
                expect(fixtureImage.prop('markers')).toEqual(props.markers);
            });

            it('passes the areas', () => {
                expect(fixtureImage.prop('areas')).toEqual(props.areas);
            });

            it('passes the edit actions', () => {
                expect(fixtureImage.props()).toEqual(jasmine.objectContaining(props.editActions));
            });

            it('passes function to update the hover state on mouse move events', () => {
                expect(wrapper.state('hover')).toEqual({ x: null, y: null });
                const pos = { x: 55, y: 66 };
                fixtureImage.prop('onMouse')(pos);
                expect(wrapper.state('hover')).toEqual(pos);
            });
        });

        describe('<ImageDetail />', () => {
            let imageDetail;

            beforeEach(() => {
                imageDetail = wrapper.find('ImageDetail');
            });

            it('renders', () => {
                expect(imageDetail.exists()).toBeTruthy();
            });

            it('passes the imageUri', () => {
                expect(imageDetail.prop('imageUri')).toEqual(props.imageUri);
            });

            it('passes the hover state', () => {
                const hover = { x: 1234, y: 9876 };
                expect(imageDetail.props())
                    .toEqual(jasmine.objectContaining({ x: null, y: null }));
                wrapper.setState({ hover });
                wrapper.update();
                imageDetail = wrapper.find('ImageDetail');
                expect(imageDetail.props())
                    .toEqual(jasmine.objectContaining(hover));
            });
        });

        describe('<FixtureGrayscalePlot />', () => {
            let plot;

            beforeEach(() => {
                plot = wrapper.find('FixtureGrayscalePlot');
            });

            it('renders', () => {
                expect(plot.exists()).toBeTruthy();
            });

            it('passes the detection pixelValues', () => {
                expect(plot.prop('pixelValues')).toEqual(grayscaleDetection.pixelValues);
            });

            it('passes the detection referenceValues', () => {
                expect(plot.prop('referenceValues')).toEqual(grayscaleDetection.referenceValues);
            });
        });

        describe('Reset all areas button', () => {
            let btn;

            beforeEach(() => {
                btn = wrapper.find('.reset-all-button');
            });

            it('renders', () => {
                expect(btn.exists()).toBeTruthy();
                expect(btn.text()).toEqual('Reset all areas');
            });

            it('calls onResetAreas when clicked', () => {
                btn.simulate('click');
                expect(onResetAreas).toHaveBeenCalled();
            });

            it('should be enabled if there are areas', () => {
                expect(btn.prop('disabled')).toBe(false);
            });
        });

        describe('Finalize button', () => {
            let btn;

            beforeEach(() => {
                btn = wrapper.find('.finalize-button');
            });

            it('renders', () => {
                expect(btn.exists()).toBeTruthy();
                expect(btn.hasClass('btn-primary')).toBeTruthy();
                expect(btn.text()).toEqual('Finalize');
            });

            it('calls onFinalize when clicked', () => {
                btn.simulate('click');
                expect(onFinalize).toHaveBeenCalled();
            });

            it('should be disabled initially', () => {
                expect(btn.prop('disabled')).toBe(true);
            });
        });
    });

    describe('No grayscale detected', () => {
        beforeEach(() => {
            wrapper = shallow(<FixtureEditor {...props} />);
        });

        describe('<FixtureGrayscalePlot />', () => {
            let plot;

            beforeEach(() => {
                plot = wrapper.find('FixtureGrayscalePlot');
            });

            it('renders', () => {
                expect(plot.exists()).toBeTruthy();
            });

            it('passes the detection pixelValues', () => {
                expect(plot.prop('pixelValues')).toEqual(null);
            });

            it('passes the detection referenceValues', () => {
                expect(plot.prop('referenceValues')).toEqual(null);
            });
        });
    });

    it('enables finalize if fixture is validFixture', () => {
        wrapper = shallow(<FixtureEditor {...props} validFixture />);
        const btn = wrapper.find('.finalize-button');
        expect(btn.prop('disabled')).toBe(false);
    });

    it('disables reset all areas button if there are no areas', () => {
        wrapper = shallow(<FixtureEditor {...props} areas={[]} />);
        const btn = wrapper.find('.reset-all-button');
        expect(btn.prop('disabled')).toBe(true);
    });
});
