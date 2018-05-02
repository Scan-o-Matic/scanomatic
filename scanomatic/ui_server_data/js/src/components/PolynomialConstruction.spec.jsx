import React from 'react';
import { shallow } from 'enzyme';

import './enzyme-setup';
import PolynomialConstruction from
    '../../src/components/PolynomialConstruction';

describe('<PolynomialConstruction />', () => {
    const onDegreeOfPolynomialChange = jasmine.createSpy('onDegreeOfPolynomialChange');
    const onFinalizeCCC = jasmine.createSpy('onFinalizeCCC');
    const props = {
        degreeOfPolynomial: 3,
        onConstruction: jasmine.createSpy('onConstruction'),
        onClearError: () => {},
        onDegreeOfPolynomialChange,
        onFinalizeCCC,
        polynomial: {
            coefficients: [42, 42, 42],
            colonies: 96,
        },
        resultsData: {
            calculated: [1, 2, 3],
            independentMeasurements: [4, 5, 6],
        },
        correlation: {
            slope: 55,
            intercept: -444,
            stderr: 0.01,
        },
        colonies: {
            independentMeasurements: [1, 2],
            pixelValues: [[1, 2], [5.5]],
            pixelCounts: [[100, 1], [44]],
            targetValues: [123, 441],
            minPixelValue: 1,
            maxPixelValue: 5.5,
            maxCount: 100,
        },
        error: 'No no no!',
    };

    beforeEach(() => {
        props.onConstruction.calls.reset();
    });


    it('should render a button to construct the polynomial', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('button.btn-construct').exists()).toBeTruthy();
        expect(wrapper.find('button.btn-construct').length).toEqual(1);
    });

    it('should render a finalize button', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('button.btn-finalize').exists()).toBeTruthy();
    });

    it('should enable the finalize button if there is a polynomial', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('button.btn-finalize').prop('disabled')).toBeFalsy();
    });

    it('should disable the finalize button if there is no polynomial', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} polynomial={null} />);
        expect(wrapper.find('button.btn-finalize').prop('disabled')).toBeTruthy();
    });

    it('should call onFinalizeCCC when the finalize button is clicked', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} polynomial={null} />);
        wrapper.find('button.btn-finalize').simulate('click');
        expect(onFinalizeCCC).toHaveBeenCalled();
    });

    it('should render a PolynomialResultsInfo', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolynomialResultsInfo').exists()).toBeTruthy();
        expect(wrapper.find('PolynomialResultsInfo').length).toEqual(1);
    });

    it('should render a PolynomialResultsPlotScatter', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolynomialResultsPlotScatter').exists())
            .toBeTruthy();
        expect(wrapper.find('PolynomialResultsPlotScatter').length).toEqual(1);
    });

    it('should render a PolynomialResultsColonyHistograms', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolynomialResultsColonyHistograms').exists())
            .toBeTruthy();
        expect(wrapper.find('PolynomialResultsColonyHistograms').length).toEqual(1);
    });

    it('should render a PolynomialConstructionError', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolynomialConstructionError').exists())
            .toBeTruthy();
        expect(wrapper.find('PolynomialConstructionError').length).toEqual(1);
    });

    it('should call onConstruction when clicked', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        wrapper.find('button.btn-construct').simulate('click');
        expect(props.onConstruction).toHaveBeenCalled();
    });

    it('should set resultsData according to props', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolynomialResultsPlotScatter')
            .prop('resultsData'))
            .toEqual(props.resultsData);
    });

    it('should set correlation according to props', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolynomialResultsPlotScatter')
            .prop('correlation'))
            .toEqual(props.correlation);
    });

    it('should set the results polynomial according to props', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolynomialResultsInfo').prop('polynomial'))
            .toEqual(props.polynomial);
    });

    it('should set the results error according to props', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolynomialConstructionError').prop('error'))
            .toEqual(props.error);
    });

    it('should set the results onClearError according to props', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolynomialConstructionError')
            .prop('onClearError')).toEqual(props.onClearError);
    });

    it('should not render any results info if there are none', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} polynomial={null} />);
        expect(wrapper.find('PolynomialResultsInfo').exists())
            .not.toBeTruthy();
        expect(wrapper.find('PolynomialResultsPlotScatter').exists())
            .toBeTruthy();
        expect(wrapper.find('PolynomialConstructionError').exists())
            .toBeTruthy();
        expect(wrapper.find('PolynomialResultsColonyHistograms').exists())
            .toBeTruthy();
    });

    it('should not render any results scatter if there are none', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} resultsData={null} />);
        expect(wrapper.find('PolynomialResultsInfo').exists())
            .toBeTruthy();
        expect(wrapper.find('PolynomialResultsPlotScatter').exists())
            .not.toBeTruthy();
        expect(wrapper.find('PolynomialConstructionError').exists())
            .toBeTruthy();
        expect(wrapper.find('PolynomialResultsColonyHistograms').exists())
            .toBeTruthy();
    });

    it('should not render any results histograms if there are none', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} colonies={null} />);
        expect(wrapper.find('PolynomialResultsInfo').exists())
            .toBeTruthy();
        expect(wrapper.find('PolynomialResultsPlotScatter').exists())
            .toBeTruthy();
        expect(wrapper.find('PolynomialResultsColonyHistograms').exists())
            .not.toBeTruthy();
        expect(wrapper.find('PolynomialConstructionError').exists())
            .toBeTruthy();
    });

    it('should not render any error if there is none', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} error={null} />);
        expect(wrapper.find('PolynomialResultsInfo').exists())
            .toBeTruthy();
        expect(wrapper.find('PolynomialConstructionError').exists())
            .not.toBeTruthy();
        expect(wrapper.find('PolynomialResultsPlotScatter').exists())
            .toBeTruthy();
        expect(wrapper.find('PolynomialResultsColonyHistograms').exists())
            .toBeTruthy();
    });

    it('should render a <select /> for the degree of the polynomial', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('select.degree').exists()).toBeTruthy();
        expect(wrapper.find('select.degree').prop('value')).toEqual(3);
    });

    it('should render options for degree 2 to 5', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        const options = wrapper.find('select.degree').find('option');
        const degrees = ['2', '3', '4', '5'];
        expect(options.map(x => x.prop('value'))).toEqual(degrees);
        expect(options.map(x => x.text())).toEqual(degrees);
    });

    it('should call onDegreeOfPolynomialChange when the selected degree changes', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        const event = { target: { value: '4' } };
        wrapper.find('select.degree').simulate('change', event);
        expect(onDegreeOfPolynomialChange).toHaveBeenCalledWith(event);
    });
});
