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

    it('should render Done as a success label', () => {
        const wrapper = shallow(<ScanningJobStatusLabel status="Done" />);
        expect(wrapper.text()).toEqual('Done');
        expect(wrapper.prop('className')).toContain('label-success');
    });

    it('should render Analysis as a default label', () => {
        const wrapper = shallow(<ScanningJobStatusLabel status="Analysis" />);
        expect(wrapper.text()).toEqual('Analysis');
        expect(wrapper.prop('className')).toContain('label-default');
    });
});
