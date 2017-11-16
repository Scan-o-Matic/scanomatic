import React from 'react';
import { shallow } from 'enzyme';

import './enzyme-setup';
import PolyConstructionButton from '../../ccc/components/PolyConstructionButton';

describe('<PolyConstructionButton />', () => {
    const props = {
        power: 5,
        onConstruction: jasmine.createSpy('onConstruction'),
    };

    beforeEach(() => {
        props.onConstruction.calls.reset();
    });

    it('should render a bootstrap button', () => {
        const wrapper = shallow(<PolyConstructionButton {...props} />);
        expect(wrapper.find('button.btn').exists()).toBeTruthy();
    });

    it('should call onConstruction when clicked', () => {
        const wrapper = shallow(<PolyConstructionButton {...props} />);
        wrapper.find('button.btn').simulate('click');
        expect(props.onConstruction).toHaveBeenCalled();

    })
});
