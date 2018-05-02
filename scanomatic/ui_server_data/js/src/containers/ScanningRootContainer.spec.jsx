import { shallow, mount } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import ScanningRootContainer from '../../src/containers/ScanningRootContainer';
import * as API from '../../src/api';
import * as helpers from '../../src/helpers';
import FakePromise from '../helpers/FakePromise';
import afterPromises from '../helpers/afterPromises';
import Duration from '../../src/Duration';


describe('<ScanningRootContainer />', () => {
    const job = {
        identifier: 'job1iamindeed',
        name: 'Test',
        scannerId: 'aha',
        duration: new Duration(5036220),
        interval: new Duration(8888),
    };
    const job2 = {
        identifier: 'job2iamevenmore',
        name: 'Test2',
        scannerId: 'ahab',
        duration: new Duration(458220),
        interval: new Duration(88888),
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
                    Object.assign({}, job, { endTime: null }),
                    Object.assign({}, job2, { endTime: null }),
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
                .toEqual([Object.assign({}, job, { endTime: null })]);
        });
    });

    describe('Resolving with a started (running or completed) job', () => {
        const startedJob = {
            name: 'Test',
            scannerId: 'aha',
            duration: new Duration(454020),
            interval: new Duration(8888),
            startTime: new Date('1980-03-23T13:00:00Z'),
        };

        beforeEach(() => {
            spyOn(API, 'getScanningJobs').and
                .returnValue(FakePromise.resolve([startedJob]));
            spyOn(helpers, 'getScannersWithOwned').and
                .returnValue(FakePromise.resolve([scanner]));
        });

        it('should compute endTime as startTime + duration', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('jobs'))
                .toEqual([Object.assign({}, startedJob, {
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

    describe('onRemoveJob', () => {
        let getScanningJobs;
        let wrapper;

        beforeEach((done) => {
            getScanningJobs = spyOn(API, 'getScanningJobs')
                .and.returnValue(Promise.resolve([job, job2]));
            spyOn(helpers, 'getScannersWithOwned')
                .and.returnValue(Promise.resolve([scanner]));
            wrapper = shallow(<ScanningRootContainer />);
            afterPromises(done);
        });

        it('calls the expected API method', () => {
            const onRemoveJob = wrapper.prop('onRemoveJob');
            spyOn(API, 'deleteScanningJob').and.returnValue(new Promise(() => {}));
            onRemoveJob('job1iamindeed');
            expect(API.deleteScanningJob).toHaveBeenCalledWith('job1iamindeed');
        });

        it('removes the job from the list', (done) => {
            const onRemoveJob = wrapper.prop('onRemoveJob');
            spyOn(API, 'deleteScanningJob').and.returnValue(new Promise(() => {}));
            onRemoveJob('job1iamindeed');
            afterPromises(() => {
                wrapper.update();
                expect(wrapper.prop('jobs')).toEqual([
                    Object.assign({}, job2, { endTime: null }),
                ]);
                done();
            });
        });

        describe('on success', () => {
            it('triggers an update of the jobs', (done) => {
                const onRemoveJob = wrapper.prop('onRemoveJob');
                spyOn(API, 'deleteScanningJob').and.returnValue(Promise.resolve());
                getScanningJobs.calls.reset();
                onRemoveJob('job1iamindeed');
                afterPromises(() => {
                    expect(getScanningJobs).toHaveBeenCalled();
                    done();
                });
            });
        });

        describe('on failure', () => {
            it('should set the error', (done) => {
                const onRemoveJob = wrapper.prop('onRemoveJob');
                const promise = Promise.reject('not good');
                spyOn(API, 'deleteScanningJob').and.returnValue(promise);
                onRemoveJob('job1iamindeed');
                afterPromises(() => {
                    wrapper.update();
                    expect(wrapper.prop('error')).toEqual('Error deleting job: not good');
                    done();
                });
            });

            it('should trigger an update of the jobs', (done) => {
                const onRemoveJob = wrapper.prop('onRemoveJob');
                spyOn(API, 'deleteScanningJob').and.returnValue(Promise.reject('not good'));
                getScanningJobs.calls.reset();
                onRemoveJob('job1iamindeed');
                afterPromises(() => {
                    expect(getScanningJobs).toHaveBeenCalled();
                    done();
                });
            });
        });
    });
});
