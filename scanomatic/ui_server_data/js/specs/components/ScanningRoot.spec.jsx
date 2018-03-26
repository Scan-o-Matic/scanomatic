
import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ScanningRoot, { getStatus, jobSorter } from '../../src/components/ScanningRoot';


describe('<ScanningRoot />', () => {
    const onNewJob = jasmine.createSpy('onNewJob');
    const onCloseNewJob = jasmine.createSpy('onCloseNewJob');
    const onStartJob = jasmine.createSpy('onStartJob');

    const props = {
        onNewJob,
        onCloseNewJob,
        onStartJob,
        error: null,
        newJob: false,
        jobs: [],
        scanners: [],
    };

    beforeEach(() => {
        onStartJob.calls.reset();
        onCloseNewJob.calls.reset();
        onNewJob.calls.reset();
    });

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
            { name: 'A', duration: { days: 0, hours: 1, minutes: 3 }, startTime: new Date('1867-03-01T06:33:12.000Z') },
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
            expect(jobPanels.last().prop('name')).toEqual('A');
            expect(jobPanels.first().prop('name')).toEqual('B');
        });

        it('couples start callback with job (first)', () => {
            const wrapper = shallow(<ScanningRoot {...props} jobs={jobs} />);
            const jobPanels = wrapper.find('ScanningJobPanel');
            jobPanels.last().prop('onStartJob')();
            expect(onStartJob)
                .toHaveBeenCalledWith(Object.assign({}, jobs[0], { status: 'Completed' }));
        });

        it('couples start callback with job (last)', () => {
            const wrapper = shallow(<ScanningRoot {...props} jobs={jobs} />);
            const jobPanels = wrapper.find('ScanningJobPanel');
            jobPanels.first().prop('onStartJob')();
            expect(onStartJob)
                .toHaveBeenCalledWith(Object.assign({}, jobs[1], { status: 'Planned' }));
        });

        it('should add statuses to the jobs', () => {
            const wrapper = shallow(<ScanningRoot {...props} jobs={jobs} />);
            const jobPanels = wrapper.find('ScanningJobPanel');
            expect(jobPanels.first().prop('status')).toEqual('Planned');
            expect(jobPanels.last().prop('status')).toEqual('Completed');
        });

        it('should pass onRemoveJob to <ScanningJobPanel/>', () => {
            const onRemoveJob = jasmine.createSpy('onRemoveJob');
            const wrapper = shallow(<ScanningRoot
                {...props}
                jobs={jobs}
                onRemoveJob={onRemoveJob}
            />);
            const jobPanels = wrapper.find('ScanningJobPanel');
            expect(jobPanels.first().prop('onRemoveJob')).toBe(onRemoveJob);
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

describe('getStatus', () => {
    it('Returns Planned if not startTime', () => {
        expect(getStatus()).toEqual('Planned');
    });

    it('Returns completed if startTime and endTime is less than now', () => {
        expect(getStatus(
            new Date('1710-05-13T06:33:12.000Z'),
            new Date('1999-12-24T06:33:12.000Z'),
            new Date('2023-04-02T06:33:12.000Z'),
        )).toEqual('Completed');
    });

    it('Returns completed if startTime and endTime is less than now', () => {
        expect(getStatus(
            new Date('1710-05-13T06:33:12.000Z'),
            new Date('2023-04-02T06:33:12.000Z'),
            new Date('1999-12-24T06:33:12.000Z'),
        )).toEqual('Running');
    });
});

describe('jobSorter', () => {
    const job0 = { status: 'Planned' };
    const job1 = { status: 'Planned' };
    const job2 = {
        status: 'Running',
        startTime: new Date('1710-05-13T06:33:12.000Z'),
        endTime: new Date('2023-04-02T06:33:12.000Z'),
    };
    const job3 = {
        status: 'Running',
        startTime: new Date('1710-05-13T06:33:12.000Z'),
        endTime: new Date('1999-12-24T06:33:12.000Z'),
    };
    const job4 = {
        status: 'Completed',
        startTime: new Date('1999-12-24T06:33:12.000Z'),
        endTime: new Date('2023-04-02T06:33:12.000Z'),
    };
    const job5 = {
        status: 'Completed',
        startTime: new Date('1710-05-13T06:33:12.000Z'),
        endTime: new Date('2023-04-02T06:33:12.000Z'),
    };

    it('should put jobs in Planned, Running, Completed order', () => {
        expect([job2, job4, job1].sort(jobSorter)).toEqual([job1, job2, job4]);
        expect([job4, job2, job1].sort(jobSorter)).toEqual([job1, job2, job4]);
        expect([job1, job4, job2].sort(jobSorter)).toEqual([job1, job2, job4]);
        expect([job1, job2, job4].sort(jobSorter)).toEqual([job1, job2, job4]);
        expect([job2, job1, job4].sort(jobSorter)).toEqual([job1, job2, job4]);
        expect([job4, job1, job2].sort(jobSorter)).toEqual([job1, job2, job4]);
        expect([job0, job1].sort(jobSorter)).toEqual([job0, job1]);
    });

    it('should sort running jobs so first to complete comes first', () => {
        expect([job2, job3].sort(jobSorter)).toEqual([job3, job2]);
    });

    it('should sort completed jobs so first started comes last', () => {
        expect([job5, job4].sort(jobSorter)).toEqual([job4, job5]);
    });
});
