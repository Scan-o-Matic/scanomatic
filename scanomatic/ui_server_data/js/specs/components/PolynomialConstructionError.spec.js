import React from 'react';
import { shallow } from 'enzyme';

import './enzyme-setup';
import PolynomialConstructionError
    from '../../src/components/PolynomialConstructionError';

describe('<PolynomialConstructionError />', () => {
    const props = {
        error: 'awesomesauce!',
        onClearError: jasmine.createSpy('onClearError'),
    };

    beforeEach(() => {
        props.onClearError.calls.reset();
    });

    it('renders an alert', () => {
        const wrapper = shallow(<PolynomialConstructionError {...props} />);
        expect(wrapper.find('div.alert').exists()).toBeTruthy();
    });

    it('doesnt render any results', () => {
        const wrapper = shallow(<PolynomialConstructionError {...props} />);
        expect(wrapper.find('div.results').exists()).not.toBeTruthy();
    });

    it('the alert displays the error', () => {
        const wrapper = shallow(<PolynomialConstructionError {...props} />);
        expect(wrapper.find('div.alert').text()).toContain(props.error);
    });

    it('the alert has a close button', () => {
        const wrapper = shallow(<PolynomialConstructionError {...props} />);
        expect(wrapper.find('div.alert').find('button').exists())
            .toBeTruthy();
    });

    it('the alert has a close button invokes onClearError', () => {
        const wrapper = shallow(<PolynomialConstructionError {...props} />);
        wrapper.find('div.alert').find('button').simulate('click');
        expect(props.onClearError).toHaveBeenCalled();
    });
});
