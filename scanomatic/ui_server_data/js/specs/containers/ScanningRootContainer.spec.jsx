import { shallow, mount } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import ScanningRootContainer from '../../src/containers/ScanningRootContainer';
import * as API from '../../src/api';
import * as helpers from '../../src/helpers';
import FakePromise from '../helpers/FakePromise';


describe('<ScanningRootContainer />', () => {
    const job = {
        identifier: 'job1iamindeed',
        name: 'Test',
        scannerId: 'aha',
        duration: {
            days: 55,
            hours: 66,
            minutes: 777,
        },
        interval: 8888,
    };
    const job2 = {
        identifier: 'job2iamevenmore',
        name: 'Test2',
        scannerId: 'ahab',
        duration: {
            days: 5,
            hours: 6,
            minutes: 77,
        },
        interval: 88888,
    };
    const scanner = {
        name: 'Kassad',
        identifier: '123asfd124kljsdf',
        owned: false,
        power: true,
    };

    describe('Jobs Request doing nothing', () => {
        beforeEach(() => {
            spyOn(API, 'getScanningJobs').and.returnValue(new FakePromise());
            spyOn(helpers, 'getScannersWithOwned')
                .and.returnValue(new FakePromise());
        });

        it('should render <ScanningRoot />', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.find('ScanningRoot').exists()).toBeTruthy();
        });

        it('should update the error on onError', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            wrapper.prop('onError')('foobar');
            wrapper.update();
            expect(wrapper.prop('error')).toEqual('foobar');
        });

        it('should pass `newJob=False`', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('newJob')).toBeFalsy();
        });

        it('should pass no error', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('error')).toBeFalsy();
        });

        it('should pass no jobs', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('jobs')).toEqual([]);
        });

        it('should set errors', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            wrapper.prop('onError')('Bad');
            wrapper.update();
            expect(wrapper.prop('error')).toEqual('Bad');
        });

        describe('New jobs', () => {
            it('should show new job', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                wrapper.prop('onNewJob')();
                wrapper.update();
                expect(wrapper.prop('newJob')).toBeTruthy();
            });

            it('should close new job', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                wrapper.setState({ newJob: true });
                wrapper.prop('onCloseNewJob')();
                wrapper.update();
                expect(wrapper.prop('newJob')).toBeFalsy();
            });

            it('should update jobs when closing', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                wrapper.prop('onNewJob')();
                expect(API.getScanningJobs).toHaveBeenCalled();
            });
        });

        it('should update jobs when mounting', () => {
            shallow(<ScanningRootContainer />);
            expect(API.getScanningJobs).toHaveBeenCalled();
        });
    });

    describe('Jobs and Scanners request resolving', () => {
        beforeEach(() => {
            spyOn(API, 'getScanningJobs').and
                .returnValue(FakePromise.resolve([job, job2]));
            spyOn(helpers, 'getScannersWithOwned').and
                .returnValue(FakePromise.resolve([scanner]));
        });

        it('should pass updated jobs', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('jobs'))
                .toEqual([
                    Object.assign({}, job, { startTime: null, endTime: null }),
                    Object.assign({}, job2, { startTime: null, endTime: null }),
                ]);
        });

        it('should pass updated scanners', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('scanners')).toEqual([scanner]);
        });

        it('should clear errors when updating', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            wrapper.setState({ error: 'test' });
            wrapper.update();
            expect(wrapper.prop('error')).toEqual('test');
            wrapper.prop('onCloseNewJob')();
            wrapper.update();
            expect(wrapper.prop('error')).toBeFalsy();
        });

        describe('Start job not doing a thing', () => {
            beforeEach(() => {
                spyOn(API, 'startScanningJob').and.returnValue(new FakePromise());
            });

            it('should deactivate button', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                const unstarted = Object.assign({}, job, { startTime: null, endTime: null });
                wrapper.prop('onStartJob')(unstarted);
                wrapper.update();
                const startingJob = Object.assign({}, unstarted);
                startingJob.disableStart = true;
                expect(wrapper.prop('jobs')).toEqual([
                    startingJob,
                    Object.assign({}, job2, { startTime: null, endTime: null }),
                ]);
            });

            it('should make an error if starting job not known', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                const unstarted = Object.assign({}, job, { startTime: null, endTime: null, identifier: 'whoami' });
                wrapper.prop('onStartJob')(unstarted);
                wrapper.update();
                expect(wrapper.prop('error')).toEqual(`UI lost job '${job.name}'`);
                expect(wrapper.prop('jobs')[0].disableStart).toBeFalsy();
            });
        });

        describe('Start Job resolving', () => {
            beforeEach(() => {
                spyOn(API, 'startScanningJob').and.returnValue(FakePromise.resolve());
            });

            it('should update the job', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                wrapper.prop('onStartJob')(job);
                expect(API.startScanningJob).toHaveBeenCalledWith(job);
            });

            it('should clear errors', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                wrapper.setState({ error: 'test' });
                wrapper.update();
                expect(wrapper.prop('error')).toEqual('test');
                wrapper.prop('onStartJob')(job);
                wrapper.update();
                expect(wrapper.prop('error')).toBeFalsy();
            });

            it('should update jobs', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                API.getScanningJobs.calls.reset();
                helpers.getScannersWithOwned.calls.reset();
                wrapper.prop('onStartJob')(job);
                expect(API.getScanningJobs).toHaveBeenCalled();
                expect(helpers.getScannersWithOwned).toHaveBeenCalled();
            });
        });

        describe('Start Job refused', () => {
            let evt = null;

            beforeEach(() => {
                evt = { target: { disabled: false } };
                spyOn(API, 'startScanningJob').and.returnValue(FakePromise.reject('Busy'));
            });

            it('should set error', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                wrapper.prop('onStartJob')(job, evt);
                wrapper.update();
                expect(wrapper.prop('error')).toEqual('Error starting job: Busy');
            });
        });
    });

    describe('Jobs request rejecting', () => {
        beforeEach(() => {
            spyOn(API, 'getScanningJobs').and
                .returnValue(FakePromise.reject('Fake'));
            spyOn(helpers, 'getScannersWithOwned').and
                .returnValue(FakePromise.resolve([scanner]));
        });

        it('should set error', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('error'))
                .toEqual('Error requesting jobs: Fake');
        });

        it('should pass scanners', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('scanners')).toEqual([scanner]);
        });
    });

    describe('Scanners request rejecting', () => {
        beforeEach(() => {
            spyOn(API, 'getScanningJobs').and
                .returnValue(FakePromise.resolve([job]));
            spyOn(helpers, 'getScannersWithOwned').and
                .returnValue(FakePromise.reject('Fake'));
        });

        it('should set error', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('error'))
                .toEqual('Error requesting scanners: Fake');
        });

        it('should pass jobs', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('jobs'))
                .toEqual([Object.assign({}, job, { startTime: null, endTime: null })]);
        });
    });

    describe('Resolving with a started (running or completed) job', () => {
        const startedJob = {
            name: 'Test',
            scannerId: 'aha',
            duration: {
                days: 5,
                hours: 6,
                minutes: 7,
            },
            interval: 8888,
            startTime: '1980-03-23T13:00:00Z',
        };

        beforeEach(() => {
            spyOn(API, 'getScanningJobs').and
                .returnValue(FakePromise.resolve([startedJob]));
            spyOn(helpers, 'getScannersWithOwned').and
                .returnValue(FakePromise.resolve([scanner]));
        });

        it('should convert startTime to a date', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('jobs'))
                .toEqual([Object.assign({}, startedJob, {
                    startTime: new Date(startedJob.startTime),
                    endTime: new Date('1980-03-28T19:07:00Z'),
                })]);
        });
    });

    it('is live updating', () => {
        spyOn(API, 'getScanningJobs').and
            .returnValue(FakePromise.resolve([]));
        spyOn(helpers, 'getScannersWithOwned').and
            .returnValue(FakePromise.resolve([]));
        jasmine.clock().install();

        mount(<ScanningRootContainer />);
        expect(API.getScanningJobs.calls.count()).toEqual(1);
        expect(helpers.getScannersWithOwned.calls.count()).toEqual(1);
        jasmine.clock().tick(5000);
        expect(API.getScanningJobs.calls.count()).toEqual(1);
        expect(helpers.getScannersWithOwned.calls.count()).toEqual(1);
        jasmine.clock().tick(5050);
        expect(API.getScanningJobs.calls.count()).toEqual(2);
        expect(helpers.getScannersWithOwned.calls.count()).toEqual(2);
        jasmine.clock().tick(10000);
        expect(API.getScanningJobs.calls.count()).toEqual(3);
        expect(helpers.getScannersWithOwned.calls.count()).toEqual(3);

        jasmine.clock().uninstall();
    });
});
