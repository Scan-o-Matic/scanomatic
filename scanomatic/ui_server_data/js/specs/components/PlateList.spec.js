import { shallow } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import PlateList from '../../ccc/components/PlateList';


describe('<PlateList />', () => {
    const props = {
        plates: [
            { name: 'my-image.tiff', id: '1M4G3' },
            { name: 'my-image.tiff', id: '1M4G3' },
            { name: 'other-image.tiff', id: '1M4G32' },
        ],
    };

    it('should render a <ul>', () => {
        const wrapper = shallow(<PlateList {...props} />);
        expect(wrapper.find('ul').exists()).toBeTruthy();
    });

    it('should render one <li /> per plate', () => {
        const wrapper = shallow(<PlateList {...props} />);
        expect(wrapper.find('ul li').length).toEqual(3);
    });

    it('should render the image name', () => {
        const wrapper = shallow(<PlateList {...props} />);
        const li = wrapper.find('ul li').at(0);
        expect(li.text()).toContain('my-image.tiff');
    });
});
