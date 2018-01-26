
import { shallow } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import ScanningRootContainer from '../../src/containers/ScanningRootContainer';
import * as API from '../../src/api';
import FakePromise from '../helpers/FakePromise';


describe('<ScanningRootContainer />', () => {
    const job = {
        name: 'Test',
        scannerId: 'aha',
        duration: {
            days: 55,
            hours: 66,
            minutes: 777,
        },
        interval: 8888,
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
            spyOn(API, 'getScanners').and.returnValue(new FakePromise());
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

        describe('Start Job', () => {
            let evt = null;

            beforeEach(() => {
                evt = { target: { disabled: false } };
                spyOn(API, 'startScanningJob').and.returnValue(FakePromise.resolve());
            });

            it('should deactivate button', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                wrapper.prop('onStartJob')(job, evt);
                expect(evt.target.disabled).toBe(true);
            });

            it('should update the job', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                wrapper.prop('onStartJob')(job, evt);
                expect(API.startScanningJob).toHaveBeenCalledWith(job);
            });

            it('should clear errors', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                wrapper.setState({ error: 'test' });
                wrapper.update();
                expect(wrapper.prop('error')).toEqual('test');
                wrapper.prop('onStartJob')(job, evt);
                wrapper.update();
                expect(wrapper.prop('error')).toBeFalsy();
            });

            it('should update jobs', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                const jobsCalls = API.getScanningJobs.calls.count();
                const scannerCalls = API.getScanners.calls.count();
                wrapper.prop('onStartJob')(job, evt);
                expect(API.getScanningJobs.calls.count()).toEqual(jobsCalls + 1);
                expect(API.getScanners.calls.count()).toEqual(scannerCalls + 1);
            });
        });

        describe('Start Job refused', () => {
            let evt = null;

            beforeEach(() => {
                evt = { target: { disabled: false } };
                spyOn(API, 'startScanningJob').and.returnValue(FakePromise.reject('Busy'));
            });

            it('should set erros', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                wrapper.prop('onStartJob')(job, evt);
                wrapper.update();
                expect(wrapper.prop('error')).toEqual('Error starting job: Busy');
            });
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
                .returnValue(FakePromise.resolve([job]));
            spyOn(API, 'getScanners').and
                .returnValue(FakePromise.resolve([scanner]));
        });

        it('should pass updated jobs', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('jobs')).toEqual([job]);
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
    });

    describe('Jobs request rejecting', () => {
        beforeEach(() => {
            spyOn(API, 'getScanningJobs').and
                .returnValue(FakePromise.reject('Fake'));
            spyOn(API, 'getScanners').and
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
            spyOn(API, 'getScanners').and
                .returnValue(FakePromise.reject('Fake'));
        });

        it('should set error', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('error'))
                .toEqual('Error requesting scanners: Fake');
        });

        it('should pass jobs', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('jobs')).toEqual([job]);
        });
    });
});
