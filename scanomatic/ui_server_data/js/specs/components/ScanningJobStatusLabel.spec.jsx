import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ScanningJobStatusLabel
    from '../../src/components/ScanningJobStatusLabel';


describe('<ScanningJobStatusLabel />', () => {
    it('should render Planned as a default label', () => {
        const wrapper = shallow(<ScanningJobStatusLabel status="Planned" />);
        expect(wrapper.text()).toEqual('Planned');
        expect(wrapper.prop('className')).toContain('label-default');
    });

    it('should render Running as an info label', () => {
        const wrapper = shallow(<ScanningJobStatusLabel status="Running" />);
        expect(wrapper.text()).toEqual('Running');
        expect(wrapper.prop('className')).toContain('label-info');
    });

    it('should render Completed as a success label', () => {
        const wrapper = shallow(<ScanningJobStatusLabel status="Completed" />);
        expect(wrapper.text()).toEqual('Completed');
        expect(wrapper.prop('className')).toContain('label-success');
    });
});

