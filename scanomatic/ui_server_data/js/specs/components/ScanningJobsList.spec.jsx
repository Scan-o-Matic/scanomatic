import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ScanningJobsList from '../../src/components/ScanningJobsList';


describe('<ScanningJobsList />', () => {
    const jobs = [
        { name: 'A' },
        { name: 'B' },
    ];

    it('renders a div', () => {
        const wrapper = shallow(<ScanningJobsList jobs={[]} />);
        expect(wrapper.find('div').exists()).toBeTruthy();
    });

    it('renders <ScanningJobPanel />:s equal to number of jobs', () => {
        const wrapper = shallow(<ScanningJobsList jobs={jobs} />);
        expect(wrapper.find('ScanningJobPanel').length).toEqual(2);
    });

    it('passes jobs-data to <ScanningJobPanel />:s', () => {
        const wrapper = shallow(<ScanningJobsList jobs={jobs} />);
        const jobPanels = wrapper.find('ScanningJobPanel');
        expect(jobPanels.first().prop('name')).toEqual('A');
        expect(jobPanels.last().prop('name')).toEqual('B');
    });
});
