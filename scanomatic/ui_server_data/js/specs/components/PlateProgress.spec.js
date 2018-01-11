import React from 'react';
import { shallow } from 'enzyme';

import './enzyme-setup';
import PlateProgress from '../../src/components/PlateProgress';

describe('<PlateProgress />', () => {
    const props = {
        now: 7,
        max: 42,
    };

    it('should render a bootstrap progress bar', () => {
        const wrapper = shallow(<PlateProgress {...props} />);
        expect(wrapper.find('div.progress').exists()).toBeTruthy();
        expect(wrapper.find('div.progress').find('div.progress-bar').exists())
            .toBeTruthy();
    });

    it('should set the progress bar width according to the props', () => {
        const wrapper = shallow(<PlateProgress {...props} />);
        expect(wrapper.find('div.progress-bar').prop('style').width)
            .toEqual('17%');
    });

    it('should set the bar text according to the props', () => {
        const wrapper = shallow(<PlateProgress {...props} />);
        expect(wrapper.find('div.progress-bar').text()).toEqual('7/42');
    });

    it('should give the progress bar a min-width so that the text is shown', () => {
        const wrapper = shallow(<PlateProgress {...props} />);
        expect(wrapper.find('div.progress-bar').prop('style').minWidth)
            .toEqual('3em');
    });
});
