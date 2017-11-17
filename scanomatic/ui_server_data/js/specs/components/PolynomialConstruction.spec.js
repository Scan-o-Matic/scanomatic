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

    it('should set the button onConstruction according to props', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('button.btn')
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
        expect(wrapper.find('PolynomialResultsInfo').prop('polynomial'))
            .toEqual(props.polynomial);
    });

    it('should set the results error according to props', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolynomialResultsInfo').prop('error'))
            .toEqual(props.error);
    });

    it('should set the results onClearError according to props', () => {
        const wrapper = shallow(<PolynomialConstruction {...props} />);
        expect(wrapper.find('PolynomialResultsInfo').prop('onClearError'))
            .toEqual(props.onClearError);
    });

    it(
        'should not render an results if there are non and there is no error',
        () => {
            const wrapper = shallow(
                <PolynomialConstruction
                    {...props}
                    error={null}
                    polynomial={null}
                />);
        }
    );
});
