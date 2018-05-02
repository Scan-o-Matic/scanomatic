
import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ScanningRoot, { getStatus, jobSorter } from '../../src/components/ScanningRoot';
import Duration from '../../src/Duration';


function makeJob(properties) {
    return Object.assign(
        {},
        {
            name: 'Test Job',
            identifier: 'job003',
            duration: new Duration(1800),
            interval: new Duration(300),
            scannerId: 'scanner025',
            startTime: undefined,
            terminationTime: undefined,
            terminationMessage: undefined,
        },
        properties,
    );
}


describe('<ScanningRoot />', () => {
    const onNewJob = jasmine.createSpy('onNewJob');
    const onCloseNewJob = jasmine.createSpy('onCloseNewJob');
    const onStartJob = jasmine.createSpy('onStartJob');

    const props = {
        error: null,
        jobs: [],
        newJob: false,
        onCloseNewJob,
        onNewJob,
        onRemoveJob: () => {},
        onStartJob,
        onStopJob: () => {},
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
            makeJob({
                duration: new Duration(1800),
                name: 'A',
                startTime: new Duration(3600).before(new Date()),
            }),
            makeJob({ name: 'B', startTime: null }),
        ];

        it('should render a list of jobs', () => {
            const wrapper = shallow(<ScanningRoot {...props} />);
            expect(wrapper.find('div.jobs-list').exists()).toBeTruthy();
        });

        it('should not render a <NewScanningJobContainer />', () => {
            const wrapper = shallow(<ScanningRoot {...props} />);
            expect(wrapper.find('NewScanningJobContainer').exists()).toBeFalsy();
        });

        it('renders <ScanningJobContainer />:s equal to number of jobs', () => {
            const wrapper = shallow(<ScanningRoot {...props} jobs={jobs} />);
            expect(wrapper.find('ScanningJobContainer').length).toEqual(2);
        });

        it('passes jobs-data to <ScanningJobContainer />:s', () => {
            const wrapper = shallow(<ScanningRoot {...props} jobs={jobs} />);
            const jobPanels = wrapper.find('ScanningJobContainer');
            expect(jobPanels.last().prop('scanningJob').name).toEqual('A');
            expect(jobPanels.first().prop('scanningJob').name).toEqual('B');
        });

        it('should add statuses to the jobs', () => {
            const wrapper = shallow(<ScanningRoot {...props} jobs={jobs} />);
            const jobPanels = wrapper.find('ScanningJobContainer');
            expect(jobPanels.first().prop('scanningJob').status).toEqual('Planned');
            expect(jobPanels.last().prop('scanningJob').status).toEqual('Completed');
        });

        it('should pass onRemoveJob to <ScanningJobContainer/>', () => {
            const onRemoveJob = jasmine.createSpy('onRemoveJob');
            const wrapper = shallow(<ScanningRoot
                {...props}
                jobs={jobs}
                onRemoveJob={onRemoveJob}
            />);
            const jobPanels = wrapper.find('ScanningJobContainer');
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
        const now = new Date('2001-01-01T01:01:01Z');
        expect(getStatus(makeJob({ startTime: null }, now))).toEqual('Planned');
    });

    it('Returns "Completed" if startTime and endTime is less than now', () => {
        const now = new Date('2001-01-01T01:01:01Z');
        expect(getStatus(
            makeJob({
                duration: new Duration(3600),
                startTime: new Date('2001-01-01T00:01:00Z'),
            }),
            now,
        )).toEqual('Completed');
    });

    it('Returns "Running" if startTime and endTime is less than now', () => {
        const now = new Date('2001-01-01T01:01:01Z');
        expect(getStatus(
            makeJob({
                duration: new Duration(3600),
                startTime: new Date('2001-01-01T01:00:00Z'),
            }),
            now,
        )).toEqual('Running');
    });

    it('Returns "Completed" if terminationTime less than now', () => {
        const now = new Date('2001-01-01T01:01:01Z');
        expect(getStatus(
            makeJob({
                duration: new Duration(3600),
                startTime: new Date('2001-01-01T01:00:00Z'),
                terminationTime: new Date('2001-01-01T01:01:00Z'),
            }),
            now,
        )).toEqual('Completed');
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
