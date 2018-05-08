import { mount } from 'enzyme';
import React from 'react';
import Markup from './Markup';
import './enzyme-setup';

describe('<Markup />', () => {
    it('Renders markup', () => {
        const txt = '__test__';
        const wrapper = mount(<Markup
            markdown={txt}
        />);
        expect(wrapper.html()).toEqual('<div class="markup"><strong>test</strong></div>');
    });
});
