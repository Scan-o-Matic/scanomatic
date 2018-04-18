import { mount } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ColonyFeatures from '../../src/components/ColonyFeatures';
import data from '../fixtures/colonyData';


describe('<ColonyFeatures />', () => {
    it('should render a <canvas />', () => {
        const wrapper = mount(<ColonyFeatures data={data} />);
        expect(wrapper.find('canvas').exists()).toBe(true);
    });
});
