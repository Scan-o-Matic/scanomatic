import React from 'react';
import { shallow } from 'enzyme';

import './enzyme-setup';
import PolynomialConstruction from
    '../../ccc/components/PolynomialConstruction';

describe('<PolynomialConstruction />', () => {
    const props = {
        onConstruction: () => {},
        onClearError: () => {},
        power: 1024,
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

    it('should render a PolyConstructionButton', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolyConstructionButton').exists()).toBeTruthy();
        expect(wrapper.find('PolyConstructionButton').length).toEqual(1);
    });

    it('should render a PolyResults', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolyResults').exists()).toBeTruthy();
        expect(wrapper.find('PolyResults').length).toEqual(1);
    });

    it('should set the button power according to props', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolyConstructionButton').prop('power'))
            .toEqual(props.power);
    });

    it('should set the button onConstruction according to props', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolyConstructionButton')
            .prop('onConstruction'))
            .toEqual(props.onConstruction);
    });

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
