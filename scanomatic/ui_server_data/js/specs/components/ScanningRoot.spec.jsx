
import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ScanningRoot from '../../src/components/ScanningRoot';


describe('<ScanningRoot />', () => {
    const onNewJob = jasmine.createSpy('onNewJob');
    const onCloseNewJob = jasmine.createSpy('onCloseNewJob');
    const props = {
        onNewJob,
        onCloseNewJob,
        error: null,
        newJob: false,
        jobs: [],
        scanners: [],
    };

    it('should render the error', () => {
        const error = 'Bad! Worse! Predictable!';
        const wrapper = shallow(<ScanningRoot {...props} error={error} />);
        expect(wrapper.text()).toContain(error);
    });

    describe('add jobb', () => {
        it('should render add job button', () => {
            const wrapper = shallow(<ScanningRoot {...props} />);
            expect(wrapper.find('button.new-job').exists()).toBeTruthy();
        });

        it('should render add job button', () => {
            const wrapper = shallow(<ScanningRoot {...props} />);
            const btn = wrapper.find('button.new-job');
            btn.simulate('click');
            expect(onNewJob).toHaveBeenCalled();
        });
    });

    describe('showing existing jobs', () => {
        const jobs = [
            { name: 'A' },
            { name: 'B' },
        ];

        it('should render a list of jobs', () => {
            const wrapper = shallow(<ScanningRoot {...props} />);
            expect(wrapper.find('div.jobs-list').exists()).toBeTruthy();
        });

        it('should not render a <NewScanningJobContainer />', () => {
            const wrapper = shallow(<ScanningRoot {...props} />);
            expect(wrapper.find('NewScanningJobContainer').exists()).toBeFalsy();
        });

        it('renders <ScanningJobPanel />:s equal to number of jobs', () => {
            const wrapper = shallow(<ScanningRoot {...props} jobs={jobs} />);
            expect(wrapper.find('ScanningJobPanel').length).toEqual(2);
        });

        it('passes jobs-data to <ScanningJobPanel />:s', () => {
            const wrapper = shallow(<ScanningRoot {...props} jobs={jobs} />);
            const jobPanels = wrapper.find('ScanningJobPanel');
            expect(jobPanels.first().prop('name')).toEqual('A');
            expect(jobPanels.last().prop('name')).toEqual('B');
        });
    });

    describe('showing new jobs', () => {
        it('should not render a <ScanningJobsList />', () => {
            const wrapper = shallow(<ScanningRoot {...props} newJob />);
            expect(wrapper.find('div.jobs-list').exists()).toBeFalsy();
        });

        it('should render a <NewScanningJobContainer />', () => {
            const wrapper = shallow(<ScanningRoot {...props} newJob />);
            expect(wrapper.find('NewScanningJobContainer').exists()).toBeTruthy();
        });

        it('should pass `onCloseNewJob` to <NewScanningJobContainer />', () => {
            const wrapper = shallow(<ScanningRoot {...props} newJob />);
            expect(wrapper.find('NewScanningJobContainer').prop('onClose'))
                .toEqual(onCloseNewJob);
        });

        it('should disable add job button', () => {
            const wrapper = shallow(<ScanningRoot {...props} newJob />);
            const btn = wrapper.find('button.new-job');
            expect(btn.prop('disabled')).toBeTruthy();
        });
    });
});
