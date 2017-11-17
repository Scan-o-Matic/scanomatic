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
        },
        data: {
            calculated: [1, 2, 3],
            independentMeasurements: [5, 4, 3],
        },
        error: 'No no no!'
    };
    
    beforeEach(() => {
        props.onConstruction.calls.reset();
    });


    it('should render a button', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('button').exists()).toBeTruthy();
        expect(wrapper.find('button').length).toEqual(1);
    });

    it('should render a PolyResults', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolyResults').exists()).toBeTruthy();
        expect(wrapper.find('PolyResults').length).toEqual(1);
    });

    it('should set the button onConstruction according to props', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('button')
            .prop('onConstruction'))
            .toEqual(props.onConstruction);
    });

    it('should call onConstruction when clicked', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        wrapper.find('button.btn').simulate('click');
        expect(props.onConstruction).toHaveBeenCalled();

    })

    it('should set the results polynomial according to props', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolyResults').prop('polynomial'))
            .toEqual(props.polynomial);
    });

    it('should set the results data according to props', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolyResults').prop('data'))
            .toEqual(props.data);
    });

    it('should set the results error according to props', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolyResults').prop('error'))
            .toEqual(props.error);
    });

    it('should set the results onClearError according to props', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolyResults').prop('onClearError'))
            .toEqual(props.onClearError);
    });
});
