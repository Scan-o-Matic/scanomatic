import React from 'react';
import { shallow } from 'enzyme';

import './enzyme-setup';
import PolynomialConstruction from
    '../../ccc/components/PolynomialConstruction';

describe('<PolynomialConstruction />', () => {
    const props = {
        onConstruction: jasmine.createSpy('onConstruction'),
        onClearError: () => {},
        polynomial: {
            power: -7,
            coefficients: [42, 42, 42],
            colonies: 96,
        },
        resultsData: {
            calculated: [1, 2, 3],
            independentMeasurements: [4, 5, 6],
        },
        error: 'No no no!'
    };

    beforeEach(() => {
        props.onConstruction.calls.reset();
    });


    it('should render a button', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('button.btn').exists()).toBeTruthy();
        expect(wrapper.find('button.btn').length).toEqual(1);
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

    it('should render a PolynomialConstructionError', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolynomialConstructionError').exists())
            .toBeTruthy();
        expect(wrapper.find('PolynomialConstructionError').length).toEqual(1);
    });

    it('should call onConstruction when clicked', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        wrapper.find('button.btn').simulate('click');
        expect(props.onConstruction).toHaveBeenCalled();
    });

    it('should set resultsData according to props', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolynomialResultsPlotScatter')
            .prop('resultsData'))
            .toEqual(props.resultsData);
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
        const wrapper = shallow(
            <PolynomialConstruction {...props} polynomial={null} />
        );
        expect(wrapper.find('PolynomialResultsInfo').exists())
            .not.toBeTruthy();
        expect(wrapper.find('PolynomialResultsPlotScatter').exists())
            .toBeTruthy();
        expect(wrapper.find('PolynomialConstructionError').exists())
            .toBeTruthy();
    });

    it('should not render any results scatter if there are none', () => {
        const wrapper = shallow(
            <PolynomialConstruction {...props} resultsData={null} />
        );
        expect(wrapper.find('PolynomialResultsInfo').exists())
            .toBeTruthy();
        expect(wrapper.find('PolynomialResultsPlotScatter').exists())
            .not.toBeTruthy();
        expect(wrapper.find('PolynomialConstructionError').exists())
            .toBeTruthy();
    });

    it('should not render any error if there is none', () => {
        const wrapper = shallow(
            <PolynomialConstruction {...props} error={null} />
        );
        expect(wrapper.find('PolynomialResultsInfo').exists())
            .toBeTruthy();
        expect(wrapper.find('PolynomialConstructionError').exists())
            .not.toBeTruthy();
        expect(wrapper.find('PolynomialResultsPlotScatter').exists())
            .toBeTruthy();
    });
});
