import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import FinalizedCCC from '../../src/components/FinalizedCCC';
import cccMetadata from '../fixtures/cccMetadata';


describe('<FinalizedCCC />', () => {
    const props = { cccMetadata };

    it('should say "Well Done!"', () => {
        const wrapper = shallow(<FinalizedCCC {...props} />);
        expect(wrapper.text()).toContain('Well Done!');
    });

    it('should show the calibration as string', () => {
        const wrapper = shallow(<FinalizedCCC {...props} />);
        expect(wrapper.text()).toContain('S. Kombuchae, Professor X');
    });

    it('should show a button to go to the home page', () => {
        const wrapper = shallow(<FinalizedCCC {...props} />);
        const btnWrapper = wrapper.find('a.btn');
        expect(btnWrapper.exists()).toBeTruthy();
        expect(btnWrapper.text()).toContain('Go to home page');
        expect(btnWrapper.prop('href')).toEqual('/home');
    });
});
